"""Which requests are an E-stop: the path alone, the path in a scope, an actual stop (SEC3-02).

The connection guard used to give priority to any request to an E-stop PATH, so a ``GET`` (405,
nothing stopped) earned the bypass of the quota and kept its connection productive. The three
functions answer three different questions; each is pinned here with the spellings that matter.
"""

from __future__ import annotations

import pytest

from backend.src.middleware.emergency_paths import (
    EMERGENCY_STOP_PATHS,
    is_emergency_stop_path,
    is_emergency_stop_request,
    is_emergency_stop_scope,
)

PATHS = sorted(EMERGENCY_STOP_PATHS)


def test_the_set_is_exactly_the_three_open_stop_routes():
    assert PATHS == [
        "/api/v2/control/emergency",
        "/api/v2/control/emergency-stop",
        "/api/v2/tractor/emergency-stop",
    ]


@pytest.mark.parametrize("path", PATHS)
def test_the_path_classifier_ignores_a_trailing_slash(path):
    assert is_emergency_stop_path(path) and is_emergency_stop_path(path + "/")


@pytest.mark.parametrize(
    "path",
    [
        "/api/v2/control/emergency_clear",  # the authenticated clear
        "/api/v2/tractor/clear-emergency",
        "/api/v2/control/emergency/extra",
        "/api/v2/control/emergencyx",
        "/api/v1/control/emergency",
        "/API/V2/CONTROL/EMERGENCY",
        "",
        "/",
    ],
)
def test_nothing_that_merely_looks_like_a_stop_path_is_one(path):
    assert not is_emergency_stop_path(path)
    assert not is_emergency_stop_scope({"path": path, "method": "POST"})
    assert not is_emergency_stop_request({"path": path, "method": "POST"})


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", ["POST", "GET", "HEAD", "PUT", "DELETE", "OPTIONS", "PATCH"])
def test_the_scope_classifier_is_about_the_path_whatever_the_method(path, method):
    assert is_emergency_stop_scope({"path": path, "method": method})


@pytest.mark.parametrize("path", PATHS)
def test_only_a_post_is_an_actual_stop(path):
    assert is_emergency_stop_request({"path": path, "method": "POST"})
    assert is_emergency_stop_request({"path": path + "/", "method": "POST"})
    for method in ("GET", "HEAD", "PUT", "DELETE", "OPTIONS", "PATCH", "post", ""):
        assert not is_emergency_stop_request({"path": path, "method": method}), method
    assert not is_emergency_stop_request({"path": path})  # no method at all


@pytest.mark.parametrize("path", PATHS)
def test_a_root_path_in_front_of_the_scope_path_is_looked_through(path):
    scope = {"path": "/lawnberry" + path, "root_path": "/lawnberry", "method": "POST"}
    assert is_emergency_stop_scope(scope) and is_emergency_stop_request(scope)
    # a root path that is not the prefix of the path does not make anything a stop
    other = {"path": "/elsewhere" + path, "root_path": "/lawnberry", "method": "POST"}
    assert not is_emergency_stop_scope(other)
    # and a root path alone never turns a non-stop path into one
    assert not is_emergency_stop_scope(
        {"path": "/lawnberry/api/v2/control/emergency_clear", "root_path": "/lawnberry"}
    )
