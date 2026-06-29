"""Enable ``python -m ontofact_nav`` to run the demonstration scenarios.

Delegates to :func:`ontofact_nav.main.main`, which parses the optional
``hospital`` / ``warehouse`` argument from the command line.
"""

from .main import main

if __name__ == "__main__":
    main()
