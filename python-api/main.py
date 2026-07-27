"""
Twilio <-> OpenAI Realtime Voice Integration
==============================================
Flow:
  1. Twilio calls POST /incoming-call when a phone call arrives.
  2. We respond with TwiML that opens a Media Stream to /media-stream.
  3. The /media-stream WebSocket bridges audio between Twilio and OpenAI Realtime.

Audio is g711 u-law 8kHz in BOTH directions (Twilio's native telephony format,
declared to OpenAI as "audio/pcmu"), so no transcoding is needed anywhere.
"""

import os
import json
import asyncio
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Realtime model. "gpt-realtime-2.1" is the current GA model and uses the GA
# event names (session.updated, response.output_audio.delta) this file handles.
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

# Audio voice for the AI response ("alloy", "echo", "shimmer", "marin", "cedar"...)
AI_VOICE = os.getenv("AI_VOICE", "alloy")

# Set DEBUG_EVENTS=1 to print every OpenAI event type (noisy, but useful)
DEBUG_EVENTS = os.getenv("DEBUG_EVENTS") == "1"

# Have the AI speak first instead of waiting in silence. Set to 0 to disable.
AI_SPEAKS_FIRST = os.getenv("AI_SPEAKS_FIRST", "1") != "0"

GREETING_INSTRUCTION = (
    "Saluda al cliente en español de forma breve y cálida: preséntate como "
    "asesor virtual de chips y planes móviles, y pregúntale en qué lo puedes "
    "ayudar. Máximo dos frases."
)

# Events we always want to see in the logs
LOG_EVENT_TYPES = {
    "error",
    "session.created",
    "session.updated",
    "response.done",
    "rate_limits.updated",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
}

# System prompt for the AI — Spanish-language SIM card sales assistant
SYSTEM_PROMPT = """
Eres un asesor de ventas virtual experto en soluciones de conectividad y telefonía móvil.

CANAL: Estás hablando por TELÉFONO. El cliente te escucha, no te lee.
- Responde SIEMPRE en español latino, con frases cortas y naturales.
- Nunca leas listas ni viñetas: menciona máximo dos opciones por turno.
- Di los precios en palabras: "cinco soles", no "S/.5".
- Mantén cada respuesta por debajo de 3 frases y termina invitando a responder.

Tus Principios de Venta:
1. Mentalidad de Valor y Consultoría: No solo despachas productos; entiendes la necesidad del cliente para ofrecerle la mejor solución de conectividad.
2. Proactividad Comercial: Siempre buscas la oportunidad de hacer Up-Selling (ofrecer una opción de plan/chip superior que le dé más beneficio al usuario) o Cross-Selling (recomendar accesorios o servicios adicionales).
3. Persuasión Natural: Tus sugerencias de productos adicionales deben sonar útiles y oportunas, nunca agresivas o invasivas. Explica siempre el *porqué* le conviene.
4. Tono de Voz: Amigable, profesional, claro y ágil.

CATÁLOGO DE PRODUCTOS Y PRECIOS
Utiliza únicamente los siguientes productos y precios de referencia para tus recomendaciones:

1. Chips y Planes Principales
- Chip Prepago Básico (5 soles):
  *Incluye: 2GB por 7 días, llamadas ilimitadas y WhatsApp.
  *Público objetivo: Quien busca algo puntual o muy económico.
- Chip Prepago Pro (10 soles):
  *Incluye: 10GB por 15 días + redes sociales libres (FB, IG, WhatsApp) + llamadas ilimitadas.
  *Uso en Up-Selling: Cuando pidan el Básico, resalta que por el doble de precio obtienen 5 veces más gigas y el doble de días.
- Plan Postpago Ilimitado / Pro (30 soles al mes):
  *Incluye: Alta velocidad de gigas, redes ilimitadas, roaming y prioridad de señal.
  *Uso en Up-Selling: Para clientes que recargan seguido o buscan despreocuparse por el consumo.

2. Productos Complementarios (Cross-Selling)
- E-SIM / SIM Virtual (adicional 5 soles): Activación inmediata digital sin necesidad de chip físico (para equipos compatibles).
- Cargador Carga Rápida 20W (35 soles): Accesorio recomendado al comprar cualquier chip o plan.
- Lector / Módem USB para SIM (25 soles): Para conectar la SIM directamente a una laptop o PC sin depender del celular.
- Mica Protectora + Funda (15 soles): Protección básica para la pantalla y el equipo.

REGLAS DE INTERACCIÓN Y TÉCNICAS DE VENTA

1. Atención a la Solicitud Inicial: Responde amablemente a la duda o pedido del cliente.
2. Estrategia de Up-Selling:
   - Si el cliente pide el Chip Básico, menciona brevemente las ventajas del Chip Pro o Plan Postpago antes de confirmar su pedido.
   - Ejemplo: "¡Claro que sí! Tengo el Chip Básico a cinco soles, pero por cinco soles más te llevas el Chip Pro con el quíntuple de gigas. ¿Te interesa?"
3. Estrategia de Cross-Selling
   - Una vez definido el chip o plan, sugiere un complemento de manera lógica antes de cerrar.
   - Ejemplo: "¿Tu equipo soporta E-SIM para activarlo de una vez sin chip físico?"
4. Cierre de Venta: Manten las opciones claras y facilita el siguiente paso (pedir datos de envío, método de pago o confirmación).
"""

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "openai_key_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_REALTIME_MODEL,
        "voice": AI_VOICE,
    }


# ---------------------------------------------------------------------------
# 1. Incoming call — return TwiML to start a Media Stream
# ---------------------------------------------------------------------------

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming_call(request: Request):
    """
    Twilio hits this endpoint when a call arrives.
    We respond with TwiML that connects the call audio to our WebSocket.
    The <Stream> URL must use wss:// with the public hostname of this server.
    """
    # Derive the public host from the request so it works on any deployment
    host = request.headers.get("host", "localhost")
    stream_url = f"wss://{host}/media-stream"

    print(f"[incoming-call] New call received. Directing stream to {stream_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>"""

    return HTMLResponse(content=twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# 2. Media stream — bridge Twilio audio <-> OpenAI Realtime
# ---------------------------------------------------------------------------

@app.websocket("/media-stream")
async def media_stream(twilio_ws: WebSocket):
    """
    This WebSocket is called by Twilio after the call connects.
    We open a second WebSocket to OpenAI Realtime and relay audio both ways.

    Twilio  →  /media-stream  →  OpenAI Realtime
    Twilio  ←  /media-stream  ←  OpenAI Realtime
    """
    await twilio_ws.accept()
    print("[media-stream] Twilio connected.")

    if not OPENAI_API_KEY:
        print("[media-stream] ERROR: OPENAI_API_KEY is not set.")
        await twilio_ws.close()
        return

    try:
        async with websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        ) as openai_ws:
            print(f"[media-stream] Connected to OpenAI Realtime: {OPENAI_REALTIME_URL}")

            await send_session_update(openai_ws)

            # --- Connection state shared by both relay directions -----------
            stream_sid = None            # Twilio stream id, needed to send audio back
            latest_media_timestamp = 0   # ms of caller audio received so far
            last_assistant_item = None   # id of the AI message currently playing
            response_start_timestamp = None
            mark_queue = []              # outstanding "mark" acks from Twilio
            greeted = False              # opening greeting already requested?

            async def relay_twilio_to_openai():
                """Read messages from Twilio and forward caller audio to OpenAI."""
                nonlocal stream_sid, latest_media_timestamp
                nonlocal last_assistant_item, response_start_timestamp

                async for raw_message in twilio_ws.iter_text():
                    message = json.loads(raw_message)
                    event = message.get("event")

                    if event == "media":
                        latest_media_timestamp = int(message["media"]["timestamp"])
                        # Twilio's payload is already base64 g711 u-law: pass through.
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": message["media"]["payload"],
                        }))

                    elif event == "start":
                        stream_sid = message["start"]["streamSid"]
                        latest_media_timestamp = 0
                        last_assistant_item = None
                        response_start_timestamp = None
                        mark_queue.clear()
                        print(f"[twilio] Stream started. streamSid={stream_sid}")

                    elif event == "mark":
                        if mark_queue:
                            mark_queue.pop(0)

                    elif event == "connected":
                        print("[twilio] Stream connected.")

                    elif event == "stop":
                        print("[twilio] Stream stopped by Twilio.")
                        break

                    else:
                        print(f"[twilio] Unhandled event: {event}")

            async def relay_openai_to_twilio():
                """Read events from OpenAI and send generated audio back to Twilio."""
                nonlocal last_assistant_item, response_start_timestamp, greeted

                async for raw_message in openai_ws:
                    response = json.loads(raw_message)
                    event_type = response.get("type")

                    if event_type in LOG_EVENT_TYPES:
                        if event_type == "error":
                            print(f"[openai] ERROR: {json.dumps(response.get('error', {}))}")
                        else:
                            print(f"[openai] {event_type}")
                    elif DEBUG_EVENTS:
                        print(f"[openai] {event_type}")

                    # Only greet once the session is confirmed, otherwise the
                    # greeting would be rendered with the default 24kHz PCM
                    # format instead of the u-law Twilio needs.
                    if event_type == "session.updated" and AI_SPEAKS_FIRST and not greeted:
                        greeted = True
                        await trigger_greeting(openai_ws)

                    # GA event name. The old beta name was 'response.audio.delta'.
                    if event_type == "response.output_audio.delta" and response.get("delta"):
                        if not stream_sid:
                            print("[openai→twilio] No streamSid yet; dropping audio chunk.")
                            continue

                        await twilio_ws.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": response["delta"]},
                        })

                        # Track which AI message is playing, so we can truncate it
                        # accurately if the caller interrupts.
                        item_id = response.get("item_id")
                        if item_id and item_id != last_assistant_item:
                            last_assistant_item = item_id
                            response_start_timestamp = latest_media_timestamp

                        await send_mark()

                    # Caller started talking over the AI -> barge-in
                    elif event_type == "input_audio_buffer.speech_started":
                        if last_assistant_item:
                            await handle_interruption()

            async def send_mark():
                """Ask Twilio to ack when this audio chunk has actually played."""
                if stream_sid:
                    await twilio_ws.send_json({
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {"name": "responsePart"},
                    })
                    mark_queue.append("responsePart")

            async def handle_interruption():
                """
                The caller spoke while the AI was talking. Twilio already has
                buffered audio queued, so we must:
                  1) tell OpenAI how much the caller actually heard (truncate)
                  2) tell Twilio to drop whatever is still queued (clear)
                """
                nonlocal last_assistant_item, response_start_timestamp

                if mark_queue and response_start_timestamp is not None:
                    elapsed_ms = latest_media_timestamp - response_start_timestamp
                    print(f"[barge-in] Caller interrupted after {elapsed_ms}ms.")

                    await openai_ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": max(elapsed_ms, 0),
                    }))
                    await twilio_ws.send_json({"event": "clear", "streamSid": stream_sid})

                    mark_queue.clear()
                    last_assistant_item = None
                    response_start_timestamp = None

            # Whichever side hangs up first ends the call: cancel the other
            # relay so we never leave an OpenAI session running after the
            # caller has gone (that would keep billing).
            tasks = [
                asyncio.create_task(relay_twilio_to_openai()),
                asyncio.create_task(relay_openai_to_twilio()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    print(f"[media-stream] Relay ended with {type(exc).__name__}: {exc}")

    except WebSocketDisconnect:
        print("[media-stream] Twilio disconnected.")
    except Exception as exc:
        print(f"[media-stream] Error: {type(exc).__name__}: {exc}")
    finally:
        print("[media-stream] Session ended.")


async def send_session_update(openai_ws):
    """
    Configure the OpenAI Realtime session (GA format):
    - audio/pcmu (g711 u-law 8kHz) in and out, matching Twilio exactly
    - output_modalities ["audio"] so the model speaks instead of writing text
    - server-side VAD so OpenAI decides on its own when to reply
    """
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": OPENAI_REALTIME_MODEL,
            "output_modalities": ["audio"],
            "instructions": SYSTEM_PROMPT,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad", "create_response": True, "interrupt_response": True},
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": AI_VOICE,
                },
            },
        },
    }
    await openai_ws.send(json.dumps(session_update))
    print("[openai] Sent session.update (GA format, audio/pcmu, server_vad).")


async def trigger_greeting(openai_ws):
    """Make the AI talk first, so the caller doesn't hear dead air."""
    await openai_ws.send(json.dumps({
        "type": "response.create",
        "response": {"instructions": GREETING_INSTRUCTION},
    }))
    print("[openai] Requested opening greeting.")
