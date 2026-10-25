"""Enumerate EVERY route of an ASGI app for the authorization tests.

The first version of the coverage test looked at ``fastapi.routing.APIRoute`` only, so a plain
Starlette ``Route``, a route of a mounted sub-application and a ``WebSocketRoute`` were all
invisible to it: a backdoor of any of those kinds left the suite green (review PR #20, mutants
M-ROUTE and M-WS). This walker yields all of them, recursing into ``Mount`` and ``Host``
sub-applications; a mount whose application exposes no routes (``StaticFiles``, an arbitrary ASGI
callable) is reported as ``opaque`` because nothing about it can be reviewed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from starlette.routing import BaseRoute, Host, Mount, Route, WebSocketRoute

MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
READING = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass(frozen=True)
class RouteInfo:
    kind: str  # "api", "route", "websocket", "mount", "other"
    path: str  # the full path, mount prefix included
    methods: frozenset[str]  # HTTP methods; empty for a websocket or a mount
    route: BaseRoute
    endpoint: Callable[..., Any] | None
    opaque: bool = False  # a mount whose application exposes no routes

    @property
    def mutating(self) -> bool:
        return bool(self.methods & MUTATING)

    @property
    def reading(self) -> bool:
        return bool(self.methods & READING)


def depends_on(dependant: Dependant, call: Callable[..., Any]) -> bool:
    return any(sub.call is call or depends_on(sub, call) for sub in dependant.dependencies)


def gated(info: RouteInfo, dependency: Callable[..., Any]) -> bool:
    """Whether the route carries ``dependency`` (only a FastAPI ``APIRoute`` can)."""
    return isinstance(info.route, APIRoute) and depends_on(info.route.dependant, dependency)


def walk(routes: list[BaseRoute], prefix: str = "") -> Iterator[RouteInfo]:
    for route in routes:
        path = prefix + getattr(route, "path", "")
        if isinstance(route, APIRoute):
            yield RouteInfo("api", path, frozenset(route.methods), route, route.endpoint)
        elif isinstance(route, WebSocketRoute):
            yield RouteInfo("websocket", path, frozenset(), route, route.endpoint)
        elif isinstance(route, Route):
            yield RouteInfo(
                "route",
                path,
                frozenset(route.methods or ()),
                route,
                getattr(route, "endpoint", None),
            )
        elif isinstance(route, (Mount, Host)):
            children = list(getattr(route, "routes", None) or [])
            mount_path = prefix + getattr(route, "path", "")
            yield RouteInfo("mount", mount_path, frozenset(), route, None, opaque=not children)
            yield from walk(children, mount_path)
        else:
            yield RouteInfo("other", path, frozenset(), route, getattr(route, "endpoint", None))


def routes_of(app: Any) -> list[RouteInfo]:
    return list(walk(list(app.routes)))
