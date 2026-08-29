# Backend-Client Protocol

Spaitra's backend owns inference, persistence, intent routing, session state,
and narration. The companion mobile client owns camera and microphone capture,
local TTS and haptics, and presentation of the state the backend sends. This
document records that boundary; it is not a claim of frontend ownership.

## Transport and authentication

HTTP is used for health, settings, item management, and image retrieval.
Socket.IO carries voice turns and the server-driven interaction state. Both
transports require the runtime API key when authentication is enabled. Keys,
live hosts, and user media must stay in runtime configuration and must not be
written to logs or committed documentation.

Production clients should use HTTPS and WSS. Socket.IO uses Engine.IO version 4
with WebSocket transport; a raw WebSocket client must implement the Socket.IO
framing and ping/pong protocol itself.

## HTTP surface used by the client

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check backend availability before opening a session |
| `GET`, `PATCH` | `/user-settings` | Read or update voice speed and performance mode |
| `GET` | `/items` | List taught memories |
| `DELETE` | `/items/<label>` | Delete a memory |
| `POST` | `/items/<label>/rename` | Rename a memory, with explicit conflict handling |
| `GET` | `/crop?scan_id=<id>&index=<n>` | Retrieve a matched crop from the bounded scan cache |

The backend also exposes direct `/remember`, `/scan`, `/find`, `/ask`,
`/feedback`, and `/retrain` routes for non-voice clients and test harnesses.
Debug routes are development-only and must not be used as product APIs.

## Client-to-server events

| Event | Payload | Client responsibility |
| --- | --- | --- |
| `chat_start` | `{}` | Ask the backend for the current listening prompt before recording |
| `chat_stop` | `{}` | Cancel a held voice turn without sending audio |
| `audio` | audio plus optional image and focal length | Send the completed turn; consume results asynchronously |
| `navigate` | `{"direction": "next"}` or `{"direction": "prev"}` | Move through the current scan result set |
| `shortcut_start` | `{"shortcut": "scan"}` | Begin the backend-owned shortcut flow |
| `shortcut_submit` | shortcut plus audio and optional image | Submit a shortcut turn without bypassing session rules |
| `shortcut_cancel` | `{"shortcut": "scan"}` | Cancel the active shortcut |

The client does not parse intent or invent state transitions. It sends captured
input and treats backend events as the source of truth.

## Server-to-client events

| Event | Meaning |
| --- | --- |
| `tts` | Narration to speak using the configured local voice speed |
| `session_state` | Authoritative mode and context after connection and turns |
| `listening`, `listening_stopped` | Start or reset the recording UI |
| `control` | Request a camera frame for scan, remember, or describe workflows |
| `action_result` | Structured result for scan, find, remember, ask, navigation, or settings |
| `transcription` | Optional text display and diagnostic context policy |
| `error` | Recoverable turn-level failure with code and message |
| `shortcut_listening`, `shortcut_ack`, `shortcut_error` | Shortcut-specific progress and failure signals |

`audio` and shortcut submissions do not return a synchronous result object.
The client listens for the event stream, stores `scan_id` from scan results for
later crop requests, and speaks only the narration sent in `tts`.

## State machine

The backend publishes one of these modes in `session_state.current_mode`:

| State | Client behavior |
| --- | --- |
| `idle` | Show the home interaction |
| `onboarding_teach` | Follow the server's teach prompt |
| `onboarding_await_scan` | Show the camera guidance for the onboarding scan |
| `awaiting_image` | Capture and attach the next requested image |
| `awaiting_location` | Record a room or location response without opening the camera |
| `awaiting_confirmation` | Record a yes/no response |
| `focused_on_item` | Show the current match and enable swipe navigation plus voice feedback |

The client must preserve the state names exactly. In focused-item mode it may
send navigation and audio, but the backend interprets feedback, rename, ask,
find, and return-home intents.

## Camera and voice boundary

The client requests camera and microphone permission, captures JPEG frames and
16 kHz speech audio, calculates focal length in pixels when available, plays
narration through platform TTS, and produces platform haptics. The backend
transcribes audio, decides when an image is required, runs the vision pipeline,
maintains onboarding and focused-item context, and returns accessible narration.

This division is implemented by the backend Socket.IO handlers under
`src/visual_memory/api/routes/voice_ws.py` and is exercised by the companion
Compose client's HTTP and Socket.IO adapters.
