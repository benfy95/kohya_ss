"""The repeated gradio/starlette HTTP_422 deprecation warning is filtered,
while unrelated warnings still surface."""

import unittest
import warnings

from kohya_gui.common_gui import suppress_known_dependency_warnings


class TestDependencyWarningFilters(unittest.TestCase):
    def test_starlette_422_warning_is_suppressed(self):
        from starlette import status
        from starlette.exceptions import StarletteDeprecationWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suppress_known_dependency_warnings()
            _ = status.HTTP_422_UNPROCESSABLE_ENTITY
        self.assertFalse(
            [w for w in caught if issubclass(w.category, StarletteDeprecationWarning)]
        )

    def test_other_warnings_still_shown(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            suppress_known_dependency_warnings()
            warnings.warn("something else is deprecated", UserWarning)
        self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()
