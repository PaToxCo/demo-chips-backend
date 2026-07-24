"""
Twilio <-> OpenAI Realtime Voice Integration
==============================================
Flow:
  1. Twilio calls POST /incoming-call when a phone call arrives.
  2. We respond with TwiML that opens a Media Stream to /media-stream.
  3. The /media-stream WebSocket bridges audio between Twilio and OpenAI Realtime.
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
OPENAI_REALTIME_URL = (
    "wss://api.openai.com/v1/realtime"
    "?model=gpt-realtime-2"
)

# System prompt for the AI — Spanish-language SIM card sales assistant
SYSTEM_PROMPT = """
Eres un asesor de ventas virtual experto en soluciones de conectividad y telefonía móvil. 


Tus Principios de Venta:
1. Mentalidad de Valor y Consultoría: No solo despachas productos; entiendes la necesidad del cliente para ofrecerle la mejor solución de conectividad.
2. Proactividad Comercial: Siempre buscas la oportunidad de hacer Up-Selling (ofrecer una opción de plan/chip superior que le dé más beneficio al usuario) o Cross-Selling (recomendar accesorios o servicios adicionales).
3. Persuasión Natural: Tus sugerencias de productos adicionales deben sonar útiles y oportunas, nunca agresivas o invasivas. Explica siempre el *porqué* le conviene.
4. Tono de Voz: Amigable, profesional, claro y ágil (ideal para chat).

CATÁLOGO DE PRODUCTOS Y PRECIOS
Utiliza únicamente los siguientes productos y precios de referencia para tus recomendaciones:

1. Chips y Planes Principales
- Chip Prepago Básico (S/.5):
  *Incluye: 2GB por 7 días, llamadas ilimitadas y WhatsApp.
  *Público objetivo: Quien busca algo puntual o muy económico.
- Chip Prepago Pro (S/.10):
  *Incluye: 10GB por 15 días + redes sociales libres (FB, IG, WhatsApp) + llamadas ilimitadas.
  *Uso en Up-Selling: Cuando pidan el Básico, resalta que por el doble de precio obtienen 5 veces más gigas y el doble de días.
- Plan Postpago Ilimitado / Pro (S/.30/mes):
  *Incluye: Alta velocidad de gigas, redes ilimitadas, roaming y prioridad de señal.
  *Uso en Up-Selling: Para clientes que recargan seguido o buscan despreocuparse por el consumo.

2. Productos Complementarios (Cross-Selling)
- E-SIM / SIM Virtual (Adicional S/.5): Activación inmediata digital sin necesidad de chip físico (para equipos compatibles).
- Cargador Carga Rápida 20W (S/.35): Accesorio recomendado al comprar cualquier chip o plan.
- Lector / Módem USB para SIM (S/.25): Para conectar la SIM directamente a una laptop o PC sin depender del celular.
- Mica Protectora + Funda (S/.15): Protección básica para la pantalla y el equipo.


REGLAS DE INTERACCIÓN Y TÉCNICAS DE VENTA

1. Atención a la Solicitud Inicial: Responde amablemente a la duda o pedido del cliente.
2. Estrategia de Up-Selling:
   - Si el cliente pide el Chip Básico, menciona brevemente las ventajas del Chip Pro o Plan Postpago antes de confirmar su pedido. 
   - Ejemplo: "¡Claro que sí! Tengo el Chip Básico a [Precio], pero por solo [Diferencia de precio] más te recomiendo el Chip Pro que te da el quíntuple de gigas y dura el doble de días. ¿Te gustaría aprovechar esa opción?"
3. Estrategia de Cross-Selling
   - Una vez definido el chip o plan, sugiere un complemento de manera lógica antes de cerrar.
   - Ejemplo: "¿Tu equipo soporta E-SIM para activarlo de una vez sin usar chip físico?" o "¿De casualidad necesitas un cargador de carga rápida para mantener tu equipo al 100%?"
4. Cierre de Venta: Manten las opciones claras y facilita el siguiente paso (ejemplo: pedir datos de envío, método de pago o confirmación).

INSTRUCCIÓN DE INICIO
Permanece a la espera del primer saludo o consulta del cliente para iniciar la atención.
"""

# Audio voice for the AI response
AI_VOICE = "alloy"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 1. Incoming call — return TwiML to start a Media Stream
# ---------------------------------------------------------------------------

@app.post("/incoming-call")
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

    stream_sid = None  # Twilio stream identifier, needed to send audio back

    # Open connection to OpenAI Realtime API
    openai_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(
            OPENAI_REALTIME_URL, additional_headers=openai_headers
        ) as openai_ws:
            print("[media-stream] Connected to OpenAI Realtime API.")

            # Send initial session configuration to OpenAI
            await send_session_update(openai_ws)

            # Run both relay directions concurrently
            await asyncio.gather(
                relay_twilio_to_openai(twilio_ws, openai_ws, lambda sid: None),
                relay_openai_to_twilio(openai_ws, twilio_ws, lambda: stream_sid),
                # We use a shared mutable dict to pass streamSid across tasks
                return_exceptions=True,
            )

    except WebSocketDisconnect:
        print("[media-stream] Twilio disconnected.")
    except Exception as exc:
        print(f"[media-stream] Error: {exc}")
    finally:
        print("[media-stream] Session ended.")


async def send_session_update(openai_ws):
    """
    Configure the OpenAI Realtime session:
    - g711_ulaw audio format (required for Twilio compatibility)
    - Server-side VAD (voice activity detection) so OpenAI knows when to respond
    - Spanish sales assistant instructions
    """
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": AI_VOICE,
            "instructions": SYSTEM_PROMPT,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        },
    }
    await openai_ws.send(json.dumps(session_update))
    print("[openai] Sent session.update (voice, instructions, g711_ulaw format).")


async def relay_twilio_to_openai(twilio_ws: WebSocket, openai_ws, set_sid):
    """
    Reads messages from Twilio and forwards audio to OpenAI.

    Twilio message types we handle:
      - 'connected'  : WebSocket handshake confirmation (no action needed)
      - 'start'      : Stream started; contains streamSid
      - 'media'      : Audio chunk (base64 g711_ulaw); forwarded to OpenAI
      - 'stop'       : Call ended
    """
    # Shared dict so we can set streamSid from this coroutine
    # and read it in the other direction coroutine
    stream_sid_holder = {"sid": None}

    try:
        async for raw_message in twilio_ws.iter_text():
            message = json.loads(raw_message)
            event = message.get("event")

            if event == "connected":
                print("[twilio] Stream connected.")

            elif event == "start":
                stream_sid = message["start"]["streamSid"]
                stream_sid_holder["sid"] = stream_sid
                # Store on the websocket state so the other task can read it
                twilio_ws.state.stream_sid = stream_sid
                print(f"[twilio] Stream started. streamSid={stream_sid}")

            elif event == "media":
                # Forward raw audio payload to OpenAI
                audio_payload = message["media"]["payload"]
                openai_event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_payload,
                }
                await openai_ws.send(json.dumps(openai_event))
                # (High-volume; comment out the line below if logs are too noisy)
                # print("[twilio→openai] Forwarded audio chunk.")

            elif event == "stop":
                print("[twilio] Stream stopped by Twilio.")
                break

            else:
                print(f"[twilio] Unknown event: {event}")

    except Exception as exc:
        print(f"[twilio→openai] Error: {exc}")


async def relay_openai_to_twilio(openai_ws, twilio_ws: WebSocket, get_sid):
    """
    Reads events from OpenAI and sends audio back to Twilio.

    OpenAI events we handle:
      - 'session.updated'         : Confirmation that session config was applied
      - 'response.audio.delta'    : Audio chunk to send back to Twilio
      - 'response.audio.done'     : OpenAI finished speaking this turn
      - 'input_audio_buffer.*'    : VAD events (logged for debugging)
      - 'error'                   : OpenAI error; logged
    """
    try:
        async for raw_message in openai_ws:
            response = json.loads(raw_message)
            event_type = response.get("type")

            if event_type == "session.updated":
                print("[openai] Session updated confirmed.")

            elif event_type == "response.audio.delta":
                # Send audio chunk back to Twilio
                audio_delta = response.get("delta", "")
                if audio_delta:
                    # Read streamSid stored by the other task
                    stream_sid = getattr(twilio_ws.state, "stream_sid", None)
                    if stream_sid:
                        twilio_payload = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_delta},
                        }
                        await twilio_ws.send_text(json.dumps(twilio_payload))
                        # (High-volume; comment out the line below if logs are too noisy)
                        # print("[openai→twilio] Forwarded audio chunk.")
                    else:
                        print("[openai→twilio] No streamSid yet; dropping audio chunk.")

            elif event_type == "response.audio.done":
                print("[openai] AI finished speaking this turn.")

            elif event_type in (
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed",
            ):
                print(f"[openai] VAD event: {event_type}")

            elif event_type == "error":
                error = response.get("error", {})
                print(f"[openai] ERROR: {error.get('message')} (code={error.get('code')})")

            else:
                # Uncomment to log every OpenAI event type during debugging:
                # print(f"[openai] Event: {event_type}")
                pass

    except Exception as exc:
        print(f"[openai→twilio] Error: {exc}")
