"""
APEX Reactive Architecture - Functional Reactive Programming Implementation

This module implements a sophisticated functional reactive programming system
with Observable streams, immutable state management, and advanced functional
programming patterns for the APEX platform.

Author: APEX Development Team
Version: 1.0.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Protocol, Union, Optional, 
    List, Dict, Any, Callable, Awaitable, Tuple,
    AsyncIterator, AsyncGenerator, Iterator
)
import asyncio
from datetime import datetime, timedelta
from collections import deque, defaultdict
import weakref
import functools
import operator

from .models import Result, Success, Failure, QueryContext, PerformanceMetrics

# Type variables
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# Functor interface
class Functor(Protocol[T]):
    """Functor type class for functional programming"""
    
    def map(self, f: Callable[[T], U]) -> 'Functor[U]':
        """Map function over functor"""
        ...

# Monad interface
class Monad(Protocol[T]):
    """Monad type class for functional programming"""
    
    def bind(self, f: Callable[[T], 'Monad[U]']) -> 'Monad[U]':
        """Bind function over monad"""
        ...
    
    def return_(self, value: U) -> 'Monad[U]':
        """Return value into monad"""
        ...

@dataclass
class Event(Generic[T]):
    """Immutable event in reactive stream"""
    value: T
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(id(datetime.utcnow())))
    
    def map(self, f: Callable[[T], U]) -> Event[U]:
        """Map function over event value"""
        return Event(
            value=f(self.value),
            timestamp=self.timestamp,
            metadata=self.metadata,
            event_id=self.event_id
        )

class Observable(Generic[T]):
    """Reactive observable with functional composition and advanced features"""
    
    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._subscribers: List[Callable[[Event[T]], None]] = []
        self._completed = False
        self._error: Optional[Exception] = None
        self._event_history: deque = deque(maxlen=1000)  # Keep last 1000 events
        self._lock = asyncio.Lock()
        self._metrics = {
            'total_events': 0,
            'total_subscribers': 0,
            'error_count': 0
        }
    
    def subscribe(self, observer: Callable[[Event[T]], None]) -> 'Subscription':
        """Subscribe to observable events with metrics tracking"""
        if self._completed:
            if self._error:
                raise self._error
            return Subscription(lambda: None)
        
        self._subscribers.append(observer)
        self._metrics['total_subscribers'] += 1
        
        return Subscription(lambda: self._unsubscribe(observer))
    
    def _unsubscribe(self, observer: Callable[[Event[T]], None]) -> None:
        """Unsubscribe observer with metrics tracking"""
        if observer in self._subscribers:
            self._subscribers.remove(observer)
            self._metrics['total_subscribers'] = max(0, self._metrics['total_subscribers'] - 1)
    
    async def emit(self, event: Event[T]) -> None:
        """Emit event to all subscribers with async safety"""
        if self._completed:
            return
        
        async with self._lock:
            # Store event in history
            self._event_history.append(event)
            self._metrics['total_events'] += 1
            
            # Emit to all subscribers
            subscribers_copy = self._subscribers[:]  # Copy to avoid modification during iteration
            
            for subscriber in subscribers_copy:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event)
                    else:
                        subscriber(event)
                except Exception as e:
                    self._metrics['error_count'] += 1
                    print(f"Error in observable {self.name}: {e}")
                    # Don't stop other subscribers on error
    
    def emit_sync(self, event: Event[T]) -> None:
        """Synchronous emit for non-async contexts"""
        if self._completed:
            return
        
        # Store event in history
        self._event_history.append(event)
        self._metrics['total_events'] += 1
        
        # Emit to all subscribers
        subscribers_copy = self._subscribers[:]
        
        for subscriber in subscribers_copy:
            try:
                subscriber(event)
            except Exception as e:
                self._metrics['error_count'] += 1
                print(f"Error in observable {self.name}: {e}")
    
    def complete(self) -> None:
        """Complete the observable"""
        self._completed = True
        self._subscribers.clear()
    
    def error(self, error: Exception) -> None:
        """Emit error and complete"""
        self._error = error
        self._completed = True
        self._subscribers.clear()
    
    # Functor implementation
    def map(self, f: Callable[[T], U]) -> 'Observable[U]':
        """Map function over observable (Functor)"""
        result = Observable[U](name=f"{self.name}.map")
        
        def mapped_observer(event: Event[T]) -> None:
            try:
                mapped_event = event.map(f)
                result.emit_sync(mapped_event)
            except Exception as e:
                result.error(e)
        
        self.subscribe(mapped_observer)
        return result
    
    # Monad implementation
    def bind(self, f: Callable[[T], 'Observable[U]']) -> 'Observable[U]':
        """Bind function over observable (Monad)"""
        result = Observable[U](name=f"{self.name}.bind")
        
        def bound_observer(event: Event[T]) -> None:
            try:
                inner_observable = f(event.value)
                inner_observable.subscribe(lambda inner_event: result.emit_sync(inner_event))
            except Exception as e:
                result.error(e)
        
        self.subscribe(bound_observer)
        return result
    
    # Advanced operators
    def filter(self, predicate: Callable[[T], bool]) -> 'Observable[T]':
        """Filter events based on predicate"""
        result = Observable[T](name=f"{self.name}.filter")
        
        def filter_observer(event: Event[T]) -> None:
            if predicate(event.value):
                result.emit_sync(event)
        
        self.subscribe(filter_observer)
        return result
    
    def take(self, count: int) -> 'Observable[T]':
        """Take only the first n events"""
        result = Observable[T](name=f"{self.name}.take({count})")
        taken = 0
        
        def take_observer(event: Event[T]) -> None:
            nonlocal taken
            if taken < count:
                result.emit_sync(event)
                taken += 1
                if taken >= count:
                    result.complete()
        
        self.subscribe(take_observer)
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get observable metrics"""
        return {
            **self._metrics,
            'name': self.name,
            'completed': self._completed,
            'has_error': self._error is not None,
            'history_size': len(self._event_history)
        }
    
    def get_latest_event(self) -> Optional[Event[T]]:
        """Get the latest event from history"""
        return self._event_history[-1] if self._event_history else None

@dataclass
class Subscription:
    """Subscription handle for cleanup"""
    _unsubscribe: Callable[[], None]
    _active: bool = True
    
    def unsubscribe(self) -> None:
        """Unsubscribe from observable"""
        if self._active:
            self._unsubscribe()
            self._active = False
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.unsubscribe()

# Reactive state management
@dataclass(frozen=True)
class State(Generic[T]):
    """Immutable state with reactive updates"""
    value: T
    version: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, new_value: T, metadata: Dict[str, Any] = None) -> 'State[T]':
        """Immutable state update"""
        return State(
            value=new_value,
            version=self.version + 1,
            timestamp=datetime.utcnow(),
            metadata={**self.metadata, **(metadata or {})}
        )

class StateManager(Generic[T]):
    """Reactive state manager with functional updates and advanced features"""
    
    def __init__(self, initial_state: T, name: str = "unnamed"):
        self.name = name
        self._state = State(value=initial_state)
        self._observable = Observable[State[T]](name=f"{name}.state")
        self._history: deque = deque(maxlen=100)  # Keep last 100 states
        self._lock = asyncio.Lock()
        
        # Emit initial state
        self._observable.emit_sync(Event(value=self._state))
        self._history.append(self._state)
    
    def get_state(self) -> State[T]:
        """Get current state"""
        return self._state
    
    def get_history(self) -> List[State[T]]:
        """Get state history"""
        return list(self._history)
    
    async def update_state(self, updater: Callable[[T], T], metadata: Dict[str, Any] = None) -> State[T]:
        """Update state with functional updater"""
        async with self._lock:
            new_value = updater(self._state.value)
            new_state = self._state.update(new_value, metadata)
            
            self._state = new_state
            self._history.append(new_state)
            
            # Emit state change
            await self._observable.emit(Event(value=new_state))
            
            return new_state
    
    def update_state_sync(self, updater: Callable[[T], T], metadata: Dict[str, Any] = None) -> State[T]:
        """Synchronous state update"""
        new_value = updater(self._state.value)
        new_state = self._state.update(new_value, metadata)
        
        self._state = new_state
        self._history.append(new_state)
        
        # Emit state change
        self._observable.emit_sync(Event(value=new_state))
        
        return new_state
    
    def subscribe(self, observer: Callable[[State[T]], None]) -> Subscription:
        """Subscribe to state changes"""
        return self._observable.subscribe(lambda event: observer(event.value))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics"""
        return {
            'name': self.name,
            'current_version': self._state.version,
            'history_size': len(self._history),
            'observable_metrics': self._observable.get_metrics()
        }

# Advanced functional utilities
def compose(*functions: Callable) -> Callable:
    """Function composition: compose(f, g, h)(x) = f(g(h(x)))"""
    def composed(x):
        for f in reversed(functions):
            x = f(x)
        return x
    return composed

def curry(func: Callable, arity: int = None) -> Callable:
    """Curry function for partial application"""
    if arity is None:
        arity = func.__code__.co_argcount
    
    def curried(*args):
        if len(args) >= arity:
            return func(*args)
        return curry(lambda *more_args: func(*(args + more_args)), arity - len(args))
    
    return curried

def partial(f: Callable, *args, **kwargs) -> Callable:
    """Partial function application"""
    return functools.partial(f, *args, **kwargs)

def pipe(value: T, *functions: Callable) -> Any:
    """Pipe value through functions: pipe(x, f, g, h) = h(g(f(x)))"""
    return compose(*functions)(value)

# Memoization with functional purity
def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """Memoize function with cache invalidation"""
    cache: Dict[str, T] = {}
    
    @functools.wraps(func)
    def memoized(*args, **kwargs):
        # Create cache key from arguments
        key = str(hash((args, tuple(sorted(kwargs.items())))))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return memoized

# Reactive patterns
class Subject(Observable[T]):
    """Subject that can both emit and subscribe to events"""
    
    def __init__(self, name: str = "subject"):
        super().__init__(name)
    
    async def next(self, value: T) -> None:
        """Emit next value"""
        await self.emit(Event(value=value))
    
    def next_sync(self, value: T) -> None:
        """Synchronous emit"""
        self.emit_sync(Event(value=value))

class BehaviorSubject(Subject[T]):
    """Subject that remembers the latest value"""
    
    def __init__(self, initial_value: T, name: str = "behavior_subject"):
        super().__init__(name)
        self._current_value = initial_value
    
    async def next(self, value: T) -> None:
        """Emit next value and update current value"""
        self._current_value = value
        await super().next(value)
    
    def next_sync(self, value: T) -> None:
        """Synchronous emit"""
        self._current_value = value
        super().next_sync(value)
    
    def get_value(self) -> T:
        """Get current value"""
        return self._current_value
