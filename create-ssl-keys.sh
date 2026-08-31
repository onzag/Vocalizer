#!/usr/bin/env bash
# Generate a self-signed SSL certificate + key for the Vocalizer WebSocket server.
# This produces cert.pem and key.pem in the current directory, valid for 365 days.
# The server loads these to serve over wss:// (TLS). Because the certificate is
# self-signed, browsers/clients must accept it once (visit https://host:8222/).
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes