import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("forest_audio.timing")

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        log_msg = f"{request.method} {request.url.path} - {process_time:.2f}ms"
        
        if process_time > 2000:
            logger.warning(f"⚠️ LATENCY WARNING: {log_msg}")
        else:
            logger.info(log_msg)
            
        response.headers["X-Process-Time"] = str(process_time)
        return response
