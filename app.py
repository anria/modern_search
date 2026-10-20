import os
import modal
from fastapi import FastAPI

# 1. Define the cloud container image and install web dependencies using uv
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "uvicorn", "python-dotenv", "rank_bm25", "sentence-transformers", "pandas", "searcharray", "pandas", "numpy", "python-dotenv", 
        "a2wsgi"  # Required to translate your Handler to a serverless context
        )
    # .add_local_file("./search.py", remote_path="/root/search.py")
    # This mounts your local main.py file so it is discoverable by Python's import system
    .add_local_python_source("search")

    .add_local_file("./search.py", remote_path="/root/search.py")
    .add_local_file("./.env", remote_path="/root/.env")
    .add_local_file("./data/cases.pkl", remote_path="/root/data/cases.pkl")
)

web_app = FastAPI()

# 2. Initialize the Modal App
app = modal.App("search", image=image)


# 3. Expose your web app (no mounts parameter required)
@app.function(image=image)
@modal.asgi_app()
def serve_web_app():
    import sys
    # Explicitly append the directory where main.py sits
    sys.path.append("/root") 
    # Inside the container, main.py is in your Python path
    # Import your custom request handler class
    from search import Handler


    from a2wsgi import WSGIMiddleware
    
    # Fully featured Mock Socket to satisfy BaseHTTPRequestHandler's read/write demands
    class WSGIMockSocket:
        def __init__(self, request_bytes):
            import io
            self._rfile = io.BytesIO(request_bytes)
            self._wfile = io.BytesIO()

        def makefile(self, mode, bufsize=0):
            if 'r' in mode:
                return self._rfile
            elif 'w' in mode or 'b' in mode:
                return self._wfile
            return self._wfile

        # Satisfies direct socket write abstractions called by wfile
        def sendall(self, data):
            self._wfile.write(data)

    def wsgi_handler_app(environ, start_response):
        content_length = int(environ.get('CONTENT_LENGTH', 0) or 0)
        body = environ['wsgi.input'].read(content_length) if content_length > 0 else b""
        
        # Build standard HTTP request string from WSGI environment
        raw_request = f"{environ['REQUEST_METHOD']} {environ['PATH_INFO']}"
        if environ.get('QUERY_STRING'):
            raw_request += f"?{environ['QUERY_STRING']}"
        raw_request += f" {environ['SERVER_PROTOCOL']}\r\n"
        
        for key, val in environ.items():
            if key.startswith('HTTP_'):
                header_name = key[5:].replace('_', '-').title()
                raw_request += f"{header_name}: {val}\r\n"
        if content_length > 0:
            raw_request += f"Content-Length: {content_length}\r\n"
        raw_request += "\r\n"
        
        # Combine headers and payload, and initialize our full-duplex mock socket
        full_input = raw_request.encode('latin1') + body
        mock_socket = WSGIMockSocket(full_input)
        
        class DummyServer:
            def __init__(self):
                self.server_name = environ['SERVER_NAME']
                self.server_port = int(environ['SERVER_PORT'])
            
        # Instantiate your custom Handler class
        # It processes request data from mock_socket and writes responses right back into it
        Handler(mock_socket, (environ['REMOTE_ADDR'], int(environ['REMOTE_PORT'])), DummyServer())
        
        # Retrieve the generated HTTP response bytes from our socket write handle
        raw_output = mock_socket._wfile.getvalue()
        
        # Split HTTP response headers from body
        try:
            headers_part, body_part = raw_output.split(b'\r\n\r\n', 1)
        except ValueError:
            headers_part, body_part = raw_output, b""
            
        header_lines = headers_part.decode('latin1').split('\r\n')
        status_line = header_lines[0]
        
        # Extract status string (e.g., "200 OK")
        status_parts = status_line.split(' ', 2)
        status_string = f"{status_parts[1]} {status_parts[2]}" if len(status_parts) > 2 else "200 OK"
        
        # Parse output headers to hand over to WSGI layer
        wsgi_headers = []
        for line in header_lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                # Skip hop-by-hop tracking values handled natively by Modal routing proxies
                if k.strip().lower() not in ['server', 'date', 'connection']:
                    wsgi_headers.append((k.strip(), v.strip()))
                
        start_response(status_string, wsgi_headers)
        return [body_part]

    return WSGIMiddleware(wsgi_handler_app)