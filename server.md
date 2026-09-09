# Vocalizer Server

The vocalizer server is a websocket based service that allows you to run the vocalizer in a server mode, so you can send JSON descriptions of the audio to generate and you will receive back a mp3 or ogg file with the generated audio. This is useful if you want to run the vocalizer on a remote server or if you want to integrate it into a web application.

## Installation

By default the server needs SSL certificates to run. You can generate self-signed certificates using the following command:

```bash
bash create-ssl-keys.sh
```

## Running the server

Simplest way to run the server is to run the following command in the root of the repository:

```bash
python server.py
```

## Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | 8222 | Port to run the server on |
| `HOST` | 0.0.0.0 | Host to run the server on |
| `DEV` | 0 | If set to 1, the server will run in development mode and use dev-secret-12345678900abcdef instead of the default secret, among other dev specific things |
| `ENABLE_UNLOAD` | 0 | If set to 1, the server will allow clients to unload the model from memory. This will cause the model not to be pre-warmed on start and enables a function to unload the model |

## Actions

All actions require a base JSON payload with the following format:

```json
{
  "action": "action_name",
  "rid": "request_id",
  ...
}
```

### upload_audio

Send this message before sending a binary into the websocket.

```json
{
  "action": "upload_audio",
  "rid": "request_id",
  "filename": "file.wav",
  "hash": "sha256 hex hash of the file",
}
```

Then send the binary data of the file, according to the answer to that rid

### render_json

Renders a JSON into an audio file. The JSON is the same as the one used in the command line version of the vocalizer.

```json
{
  "action": "render_json",
  "rid": "request_id",
  "payload": { ... },
  "format": "mp3" // or "ogg"
}
```

Specify files by the filename that was given in the upload_audio action.

### ping

Just pings the server to check if it's alive. The server will respond with a pong message.

```json
{
  "action": "ping",
  "rid": "request_id"
}
```

### Events

## pong

The server will respond to a ping with a pong message.

```json
{
  "type": "pong",
  "rid": "request_id"
}
```

## error

The server will respond with an error message if something goes wrong.

```json
{
  "type": "error",
  "rid": "request_id",
  "message": "error message"
}
```

## upload_audio_skip

The server will respond with an upload_audio_skip message if the file has already been uploaded and the hash matches.

```json
{
  "type": "upload_audio_skip",
  "rid": "request_id"
}
```

## upload_audio_proceed

The server will respond with an upload_audio_proceed message if the file needs to be uploaded.

```json
{
  "type": "upload_audio_proceed",
  "rid": "request_id"
}
```

## queued

The server will respond with a queued message if the request has been queued for processing.

```json
{
  "type": "queued",
  "position": 1,
  "rid": "request_id"
}
```

## render_start

The server will respond with a render_start message when the rendering has started.

```json
{
  "type": "render_start",
  "rid": "request_id"
}
```

## binary event (not JSON)

After a render_start event the server will slowly stream the binary data of the rendered audio file. The client should read the binary data until the connection is closed or until a new JSON message is received.

## render_done

The server will respond with a render_done message when the rendering has finished.

```json
{
  "type": "render_done",
  "rid": "request_id"
}
```

## model_loaded

The server will respond with a model_loaded message when the model has been loaded into memory.

```json
{
  "type": "model_loaded",
  "rid": "request_id"
}
```

## model_unloaded

The server will respond with a model_unloaded message when the model has been unloaded from memory.

```json
{
  "type": "model_unloaded",
  "rid": "request_id"
}
```

## Special actions (only available if `ENABLE_UNLOAD` is set to 1)

### unload_model

Unloads the model from memory.

```json
{
  "action": "unload_model",
  "rid": "request_id"
}
```

## load_model

Reloads the model into memory. While the model is unloaded, `render_json` returns an error until this action completes.

```json
{
  "action": "load_model",
  "rid": "request_id"
}
```
