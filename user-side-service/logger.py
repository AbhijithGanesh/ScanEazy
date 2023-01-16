import logging
from rich.logging import RichHandler


FORMAT = "%(message)s"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[RichHandler(tracebacks_suppress=[])],
)


class Logger:
    def __init__(
        self,
        name,
        message_format="%(message)s",
        date_format="[%x]",
    ):
        self.log = logging.getLogger(name)
        self.log.__format__ = message_format
        self.log.__date_format__ = date_format

    def stringify(func):
        def inner(self, method_type: str, *msg: object, sep=" "):
            nmsg = []
            for v in msg:
                if not isinstance(v, str):
                    v = str(v)
                nmsg.append(v)
            return func(self, method_type, *nmsg, sep=sep)

        return inner

    @stringify
    def logutil(self, method_type: str, *msg: object, sep=" ") -> None:
        func = getattr(self.log, method_type, None)
        if not func:
            raise AttributeError(f"Logger has no method {method_type}")
        return func(sep.join(msg), stacklevel=4)

    def info(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("info", *msg, sep=sep)

    def warning(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("warning", *msg, sep=sep)

    def error(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("error", *msg, sep=sep)

    def critical(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("critical", *msg, sep=sep)


logger = Logger(__name__)
