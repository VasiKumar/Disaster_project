from app.config import settings


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)