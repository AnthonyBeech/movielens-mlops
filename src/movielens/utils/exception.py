class ModelError(Exception):
    """Exception raised for errors in the model."""

    def __init__(self, message: str = "An error occurred in the model.", error_code: int | None = None) -> None:
        """Init."""
        super().__init__(message)
        self.error_code = error_code

    def __str__(self) -> str:
        """Print of model error."""
        if self.error_code is not None:
            return f"{self.__class__.__name__} (Code {self.error_code}): {self.args[0]}"
        return f"{self.__class__.__name__}: {self.args[0]}"
