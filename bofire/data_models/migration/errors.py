class MigrationError(Exception):
    pass


class UnrecoverablePayloadError(MigrationError):
    def __init__(self, *, payload_type: str, reason: str, hint: str = ""):
        self.payload_type = payload_type
        self.reason = reason
        self.hint = hint
        msg = f"{payload_type}: {reason}"
        if hint:
            msg += f" Hint: {hint}"
        super().__init__(msg)


class UnknownVersionError(MigrationError):
    pass
