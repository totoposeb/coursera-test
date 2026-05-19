from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Generator, Generic, Iterator, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Direction(Enum):
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()

    def opposite(self) -> Direction:
        opposites = {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
        }
        return opposites[self]


class Color(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def blend(self, other: Color, ratio: float = 0.5) -> tuple[int, int, int]:
        r = int(self.value[0] * (1 - ratio) + other.value[0] * ratio)
        g = int(self.value[1] * (1 - ratio) + other.value[1] * ratio)
        b = int(self.value[2] * (1 - ratio) + other.value[2] * ratio)
        return (r, g, b)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __repr__(self) -> str:
        return f"Point({self.x:.2f}, {self.y:.2f})"


@dataclass
class Inventory:
    items: list[str] = field(default_factory=list)
    capacity: int = 10

    def add(self, item: str) -> bool:
        if len(self.items) >= self.capacity:
            return False
        self.items.append(item)
        return True

    def remove(self, item: str) -> bool:
        try:
            self.items.remove(item)
            return True
        except ValueError:
            return False

    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    def __len__(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------------
# Abstract base class + subclasses
# ---------------------------------------------------------------------------

class Shape(ABC):
    color: Color = Color.WHITE

    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

    def describe(self) -> str:
        return (
            f"{type(self).__name__}: "
            f"area={self.area():.3f}, perimeter={self.perimeter():.3f}"
        )


class Circle(Shape):
    def __init__(self, radius: float, color: Color = Color.RED) -> None:
        self.radius = radius
        self.color = color

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    def scale(self, factor: float) -> Circle:
        return Circle(self.radius * factor, self.color)


class Rectangle(Shape):
    def __init__(self, width: float, height: float, color: Color = Color.BLUE) -> None:
        self.width = width
        self.height = height
        self.color = color

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    @property
    def is_square(self) -> bool:
        return math.isclose(self.width, self.height)

    def rotate(self) -> Rectangle:
        return Rectangle(self.height, self.width, self.color)


class Triangle(Shape):
    def __init__(self, a: float, b: float, c: float) -> None:
        if not self._valid(a, b, c):
            raise ValueError(f"Invalid triangle sides: {a}, {b}, {c}")
        self.a, self.b, self.c = a, b, c

    @staticmethod
    def _valid(a: float, b: float, c: float) -> bool:
        return a + b > c and b + c > a and a + c > b

    def area(self) -> float:
        s = self.perimeter() / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    def perimeter(self) -> float:
        return self.a + self.b + self.c

    @classmethod
    def equilateral(cls, side: float) -> Triangle:
        return cls(side, side, side)


# ---------------------------------------------------------------------------
# Generic stack
# ---------------------------------------------------------------------------

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._data: list[T] = []

    def push(self, item: T) -> None:
        self._data.append(item)

    def pop(self) -> T:
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> T:
        if not self._data:
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(reversed(self._data))


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def clamp(minimum: float, maximum: float):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return max(minimum, min(maximum, result))
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


def memoize(func):
    cache: dict = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.__name__ = func.__name__
    wrapper.cache = cache
    return wrapper


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

@memoize
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@clamp(0.0, 1.0)
def normalize(value: float, low: float, high: float) -> float:
    if math.isclose(low, high):
        return 0.0
    return (value - low) / (high - low)


def primes_up_to(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i, v in enumerate(sieve) if v]


def flatten(nested: list) -> Generator:
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def chunk(seq: list[T], size: int) -> Generator[list[T], None, None]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def running_average(numbers: list[float]) -> list[float]:
    result, total = [], 0.0
    for i, n in enumerate(numbers, start=1):
        total += n
        result.append(total / i)
    return result


# ---------------------------------------------------------------------------
# Simple simulation
# ---------------------------------------------------------------------------

class Walker:
    def __init__(self, start: Point = Point(0, 0), seed: int | None = None) -> None:
        self.position = start
        self.history: list[Point] = [start]
        self._rng = random.Random(seed)

    def step(self) -> Direction:
        direction = self._rng.choice(list(Direction))
        delta = {
            Direction.NORTH: Point(0, 1),
            Direction.SOUTH: Point(0, -1),
            Direction.EAST: Point(1, 0),
            Direction.WEST: Point(-1, 0),
        }[direction]
        self.position = self.position + delta
        self.history.append(self.position)
        return direction

    def walk(self, steps: int) -> list[Direction]:
        return [self.step() for _ in range(steps)]

    @property
    def total_distance(self) -> float:
        return sum(
            a.distance_to(b) for a, b in zip(self.history, self.history[1:])
        )

    @property
    def displacement(self) -> float:
        return self.history[0].distance_to(self.position)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Shapes
    shapes: list[Shape] = [
        Circle(5, Color.RED),
        Rectangle(4, 6, Color.BLUE),
        Triangle.equilateral(3),
    ]
    print("=== Shapes ===")
    for shape in shapes:
        print(" ", shape.describe())

    # Generic stack
    print("\n=== Stack ===")
    stack: Stack[int] = Stack()
    for value in [10, 20, 30]:
        stack.push(value)
    print(f"  top={stack.peek()}, size={len(stack)}")
    print(f"  popped={stack.pop()}, size={len(stack)}")

    # Fibonacci
    print("\n=== Fibonacci ===")
    fibs = [fibonacci(n) for n in range(10)]
    print(f"  {fibs}")
    print(f"  cache size: {len(fibonacci.cache)}")

    # Primes
    print("\n=== Primes up to 50 ===")
    print(f"  {primes_up_to(50)}")

    # Random walk
    print("\n=== Random Walk (20 steps) ===")
    walker = Walker(seed=42)
    walker.walk(20)
    print(f"  final position : {walker.position}")
    print(f"  total distance : {walker.total_distance:.2f}")
    print(f"  displacement   : {walker.displacement:.2f}")

    # Color blend
    print("\n=== Color Blend ===")
    blended = Color.RED.blend(Color.BLUE, 0.5)
    print(f"  RED + BLUE (50%) = {blended}")

    # Normalize
    print("\n=== Normalize (clamped to [0, 1]) ===")
    values = [-10, 0, 50, 100, 150]
    for v in values:
        print(f"  normalize({v}, 0, 100) = {normalize(v, 0, 100):.2f}")

    # Flatten + chunk
    print("\n=== Flatten & Chunk ===")
    nested = [[1, 2, [3, 4]], [5, [6, [7]]]]
    flat = list(flatten(nested))
    print(f"  flat   : {flat}")
    print(f"  chunks : {list(chunk(flat, 3))}")


if __name__ == "__main__":
    main()
