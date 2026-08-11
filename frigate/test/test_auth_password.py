from frigate.api.auth import validate_password_strength


def test_password_minimum_length_is_six_characters():
    assert validate_password_strength("123456") == (True, None)
    assert validate_password_strength("12345") == (
        False,
        "Password must be at least 6 characters long",
    )
