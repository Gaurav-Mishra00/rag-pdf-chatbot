"""Safe, actionable errors that must never be stored as assistant answers."""


class GenerationError(RuntimeError):
    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.status_code = status_code


def generation_error(exc: Exception) -> GenerationError:
    # Provider adapters may wrap the original HTTP error.
    causes = []
    current = exc
    while current is not None and id(current) not in {id(e) for e in causes}:
        causes.append(current)
        current = current.__cause__ or current.__context__
    description = " ".join(f"{type(e).__name__} {e}" for e in causes).lower()
    codes = {str(getattr(e, "status_code", getattr(e, "code", ""))) for e in causes}
    if "timeout" in description or "timed out" in description or "deadline" in description:
        return GenerationError(
            "The AI service timed out. Check the server's connection to the configured AI provider and try again.", 504
        )
    if "429" in codes or "resource_exhausted" in description or "quota" in description:
        return GenerationError("The AI provider's quota or rate limit was reached. Check your provider quota and try again later.", 503)
    if codes & {"401", "403"} or "api_key_invalid" in description:
        return GenerationError("The AI provider rejected the server's credentials. Check the provider API key in .env.")
    if "404" in codes or "not_found" in description:
        return GenerationError("The configured AI model is unavailable. Check LLM_MODEL_NAME and your provider's model access in .env.")
    if codes & {"500", "502", "503", "504"}:
        return GenerationError("The AI provider returned a server error. Try again or configure another available model in LLM_MODEL_NAME.")
    if "connect" in description:
        return GenerationError("Cannot connect to the AI service. Check the server's internet connection and proxy settings.", 503)
    return GenerationError("Response generation failed. Check the server log for the underlying error.")
