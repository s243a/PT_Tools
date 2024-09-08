from typing import Callable, TypeVar, Generic, Iterator, Iterable, Union, List, Optional, Any, Generator
from typing import Dict, Tuple
from itertools import islice, count
#from lamtest import myfunc
import inspect
import traceback
import functools
import warnings
import argparse
import logging
import tempfile
import os
from collections.abc import Iterator, Iterable
import itertools
from itertools import islice

logger = logging.getLogger(__name__)
def lamtest(lambda_func):
    source_lines, _ = inspect.getsourcelines(lambda_func)
    return ''.join(source_lines)


T = TypeVar('T')  # Type of the input elements
R = TypeVar('R')  # Type of the output elements
I = TypeVar('I', bound=Iterable)  # Type of the input iterable
O = TypeVar('O')  # Type of the output collection

def experimental(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is experimental and may change in future versions",
            FutureWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper

class LazySequence:
    def __init__(self, iterator, underlying):
        self.iterator = iterator
        self.underlying = underlying

    def __iter__(self):
        yield from self.iterator

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.underlying[i] for i in range(*index.indices(len(self.underlying)))]
        return self.underlying[index]

    def __len__(self):
        return len(self.underlying)

    def __repr__(self):
        return f"LazySequence({self.underlying})"
class Box(Generic[I, T]):
    def __init__(self, 
                 value: I,
                 wrap: Optional[Callable[[Any, R, Optional[int]], None]] = None,
                 init: Optional[Callable[[], Any]] = None,
                 mutable: bool = False):
        self.value = value
        self.wrap = wrap
        self.mutable = mutable
        self.init = init

    def unBox(self) -> I:
        if self.mutable:
            return self.value
        else:
            raise Exception("Cannot unbox immutable object")

    def __set_wrap__(self, 
        wrap: Optional[Callable[[Any, R, Optional[int]], None]],
        mutable: bool = True) -> Callable[[Any, R, Optional[int]], None]:
        if wrap is not None:
            logger.debug("setting custom wrap")
            return wrap    
        elif self.wrap is not None:
            logger.debug("Using Box's wrap function")
            return self.wrap
        elif self.mutable and mutable:
            logger.debug("Using default mutable wrap function")
            return lambda coll, item, index: coll.__setitem__(index, item) if hasattr(coll, '__setitem__') else setattr(coll, f'item_{index}', item)
        else:
            return lambda coll, item, _: coll.append(item) if hasattr(coll, 'append') else None


    def __set_out_collection__(self,
        out_collection: Optional[O],
        mutable: bool = True) -> Any:

        if out_collection is not None:
            return out_collection
        elif self.mutable and mutable:
            return self.value
        elif self.init is not None:
            return self.init()
        else:
            return []

    def map(self, 
            f: Callable[[T], R], 
            wrap: Optional[Callable[[Any, R, Optional[int]], None]] = None,
            out_collection: Optional[O] = None,
            mutable: bool = True
    ) -> 'Box[O, R]':
        wrap = self.__set_wrap__(wrap,mutable=mutable)
        out_collection = self.__set_out_collection__(out_collection, mutable=mutable)    
        #print("wrap="+inspect.getsource(wrap))
        #print("out_collection="+str(out_collection))
        def lazy_map() -> Iterator[R]:
            if isinstance(self.value, dict):
                for k, v in self.value.items():
                    result = f(k)
                    if self.mutable and mutable and out_collection is self.value:
                        wrap(out_collection, result[1], result[0])
                    yield result
            else:
                for i, item in enumerate(self.value):
                    result = f(item)
                    if self.mutable and mutable and out_collection is self.value:
                        wrap(out_collection, result, i)
                    yield result

        iterator = lazy_map()
        if self.mutable and mutable and out_collection is self.value:
            # For mutable in-place modification, we return a Box with the same value
            # but wrap it in a LazySequence to preserve lazy evaluation
            logger.debug("Default lazy_map for mutable objects")
            return Box(LazySequence(iterator, out_collection), 
                       wrap=wrap, init=self.init, mutable=True)
        else:
            # For immutable or new collections, we return a Box with the iterator
            logger.debug("Default lazy_map for immutable objects")
            return Box(iterator, wrap=wrap, init=self.init, mutable=mutable)

    def __iter__(self):
        if isinstance(self.value, LazySequence):
            return iter(self.value)
        return iter(self.value)

    def __repr__(self) -> str:
        if isinstance(self.value, (Iterator, map)):
            return f"Box(<lazy>)"
        return f"Box({self.value})"
    #def strict(self, verbose=True):
    #    """
    #    Iterate through the entire collection, making it concrete.
    #    If verbose is True, print each item.
    #    """
    #    iterator = iter(self.value)
    #    try:
    #        for item in iterator:
    #            if verbose:
    #                print(item)
    #    except Exception as e:
    #        print(f"Error during strict evaluation: {e}")
    #    return self

    def take(self, N: int = 0) -> Tuple[List[T], 'Box[I, T]']:
        """
        Return a tuple containing:
        1. A list of the first N items from the collection (or all items if N=0).
        2. A new Box with the remaining items.
        
        If N is 0 or larger than the collection size, all items will be in the first element,
        and the second element will be an empty Box.
        """
        iterator = iter(self.value)
        if N == 0:
            taken = list(iterator)
        else:
            taken = list(islice(iterator, N))
        return taken, Box(iterator, wrap=self.wrap, init=self.init, mutable=self.mutable)
    @experimental
    def forked_take(self, N: int) -> Tuple[List[T], 'Box[I, T]']:
        """
        EXPERIMENTAL: This method is for future development and may change.

        Takes N items from the stream, returning them as a list along with a new Box
        containing the rest of the stream.

        This creates a fork in the stream, allowing for Haskell-like list operations.
        """
        iterator = iter(self.value)
        taken = list(itertools.islice(iterator, N))
        rest = Box(iterator, wrap=self.wrap, init=self.init, mutable=self.mutable)
        return taken, rest
    def to_list(self) -> List[T]:
        """
        Convert the Box contents to a list.
        
        Returns:
            List[T]: A list containing all elements from the Box.
        
        Raises:
            TypeError: If the Box contents are not iterable.
        """
        if isinstance(self.value, list):
            return self.value
        try:
            return list(iter(self.value))
        except TypeError:
            raise TypeError(f"Cannot convert to list: {type(self.value)} object is not iterable")
    def to_dict(self):
        out={}
        self.strict()
        print(f"self.value={self.value}")
        if isinstance(self.value, dict):
            for k, v in self.value.items():
                print(f"out[{k}]=[{v}]")
                out[k]=v
        else:
            for i, item in enumerate(self.value):
                out[i]=item

        return out
    def seek(self, N: int = 0) -> Tuple[bool, 'Box[I, T]']:
        """
        Advance the iterator N times (or to the end if N=0).
        
        Returns a tuple containing:
        1. A boolean indicating whether the end of the iterator was reached (True if EOF).
        2. A new Box with the remaining items.
        
        If N is 0 or larger than the collection size, the second element will be an empty Box.
        """
        iterator = iter(self.value)
        reached_end = False
        if N == 0:
            # Consume the entire iterator
            consumed = sum(1 for _ in iterator)
            reached_end = True
        else:
            consumed = 0
            for _ in range(N):
                try:
                    next(iterator)
                    consumed += 1
                except StopIteration:
                    reached_end = True
                    break
        
        return reached_end, Box(iterator, wrap=self.wrap, init=self.init, mutable=self.mutable)

    

    def strict(self, verbose=True):
        """
        Iterate through the entire collection, making it concrete.
        If verbose is True, print each item.
        """
        if isinstance(self.value, (Iterator, map, LazySequence)):
            concrete_value, item_type = self.__infer_type_and_create_concrete_value()
            try:
                for item in self.value:
                    self.__add_to_concrete_value(concrete_value, item, item_type)
                    if verbose:
                        print(item)
                self.value = concrete_value
            except Exception as e:
                print(f"Error during strict evaluation: {e}")
        elif verbose:
            for item in self.value:
                print(item)
        return self

    def __infer_type_and_create_concrete_value(self):
        if self.init is not None:
            return self.init(), None
        
        # Try to peek at the first item without consuming the iterator
        try:
            first_item = next(iter(self.value))
            # Put the item back into an iterator
            self.value = itertools.chain([first_item], self.value)
        except StopIteration:
            # Iterator is empty
            return [], None

        if isinstance(first_item, tuple) and len(first_item) == 2:
            # Likely a key-value pair, assume it's a dictionary
            return {}, dict
        elif isinstance(first_item, Iterable) and not isinstance(first_item, str):
            # It's some other kind of iterable, assume it's a list of lists
            return [], list
        else:
            # Default to a list
            return [], list

    def __add_to_concrete_value(self, concrete_value, item, item_type):
        if self.wrap is not None:
            if item_type == dict:
                self.wrap(concrete_value, item[1], item[0])
            else:
                self.wrap(concrete_value, item, len(concrete_value))
        elif item_type == dict:
            concrete_value[item[0]] = item[1]
        elif isinstance(concrete_value, list):
            concrete_value.append(item)
        else:
            setattr(concrete_value, f'item_{len(concrete_value)}', item)  
#################################### Test the function ########################
def double(x: int) -> int:
    return x * 2
def test_file_stream():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        # Write some data to the file
        temp_file.write("1\n2\n3\n4\n5\n")
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        # Function to read from file
        def file_reader(filename):
            with open(filename, 'r') as file:
                for line in file:
                    yield int(line.strip())

        # Create a Box with the file reader
        file_box = Box(file_reader(temp_file_name))
        
        print("file_box (before operations):")
        print(file_box)  # This should show Box(<lazy>)

        # Map operation to double each number
        doubled_file = file_box.map(double)
        
        print("doubled_file (before evaluation):")
        print(doubled_file)  # This should also show Box(<lazy>)

        # Evaluate and print the result
        result = list(doubled_file)
        print("Result of doubled_file:")
        print(result)  # This should print [2, 4, 6, 8, 10]

        # Create a new Box for testing take() method
        new_file_box = Box(file_reader(temp_file_name))
        new_doubled_file = new_file_box.map(double)

        # Test taking first 3 items
        taken, rest = new_doubled_file.take(3)
        print("First 3 items:")
        print(taken)  # This should print [2, 4, 6]

        # Test remaining items
        remaining = list(rest)
        print("Remaining items:")
        print(remaining)  # This should print [8, 10]

    finally:
        # Clean up: remove the temporary file
        os.unlink(temp_file_name)

def test():

    # Example usage

    tests={
     "Immutable_list" : False,
     "Mutable_list" :False,
     "Dictionary" : False,
     "Dictionary_with_custom_wrap" : False,
     "Custom_collection" :False,
     "Lazy_integers" : False,
     "Generator" : False,
     "Infinite_stream" : False,
     "Lazy_iterator" : False, 
     "Custom_iterable" : False,
     "Single_value" : False,
     "Iterable_but_not_iterator" : False,
     "File_stream" : False,
     "Strict_conversion": False,
     "To_dict_conversion": False,
     "Forked_take": True     
    }
    for test_name, enabled in tests.items():
        if enabled:
            print(f"\n# Running test: {test_name}")
            try:   
                if test_name == "Immutable_list":
                    print("immutable_box = Box([1, 2, 3], mutable=False)")
                    immutable_box = Box([1, 2, 3], mutable=False)
                    doubled_immutable = immutable_box.map(double)
                    print(f"doubled_immutable={doubled_immutable}")
                    doubled_immutable_to_list = doubled_immutable.to_list()
                    print(f"doubled_immutable.to_list()={doubled_immutable_to_list}")
                    assert doubled_immutable_to_list == [2, 4, 6], "Unexpected result for immutable list"

                elif test_name == "Mutable_list":
                    mutable_box = Box([1, 2, 3], mutable=True)
                    print("mutable_box = Box([1, 2, 3], mutable=True)")
                    doubled_mutable = mutable_box.map(double)
                    print(f"doubled_mutable={doubled_mutable.to_list()}")# Output: Box([2, 4, 6])
                    print(f"mutable_box.unBox()={mutable_box.unBox()}")  # Output: [2, 4, 6]
                    print(f"mutable_box.to_list()={mutable_box.to_list()}")  # Output: [2, 4, 6]


                if test_name == "Dictionary":
                    dict_box = Box({'a': 1, 'b': 2, 'c': 3}, mutable=True)
                    print("Original dict_box:")
                    print(dict_box.unBox())
                    
                    doubled_dict = dict_box.map(lambda k: (k, double(dict_box.value[k])))
                    print("doubled_dict (before iteration):")
                    print(doubled_dict)
                    
                    print("dict_box after mapping (before iteration):")
                    print(dict_box.unBox())
                    
                    print("Iterating over doubled_dict:")
                    for k, v in doubled_dict:
                        print(f"{k}: {v}")

                    print("Final state of dict_box:")
                    print(dict_box.unBox())

                elif test_name == "Dictionary_with_custom_wrap":
                    def custom_dict_wrap(d, item, key):
                        print(f"Custom wrap called with key: {key}, value: {item}")
                        d[key] = item

                    dict_box = Box({'a': 1, 'b': 2, 'c': 3}, 
                                   wrap=custom_dict_wrap,
                                   init=dict,
                                   mutable=True)
                    print("Original dict_box:")
                    print(dict_box.unBox())
                    
                    doubled_dict = dict_box.map(lambda k: (k, double(dict_box.value[k])))
                    print("doubled_dict (before iteration):")
                    print(doubled_dict)
                    
                    print("dict_box after mapping (before iteration):")
                    print(dict_box.unBox())
                    
                    print("Iterating over doubled_dict:")
                    list(doubled_dict)  # Force iteration

                    print("Final state of dict_box:")
                    print(dict_box.unBox())               


                elif test_name == "Custom_collection":
                    class CustomCollection:
                        def __init__(self):
                            self.data = []
                        def add(self, item):
                            self.data.append(item)
                        def __repr__(self):
                            return f"CustomCollection({self.data})"

                    custom_box = Box([1, 2, 3])
                    custom_collection = CustomCollection()
                    def custom_wrap(coll, item, _):
                        coll.add(item)
                    doubled_custom = custom_box.map(
                        double, 
                        wrap=custom_wrap, 
                        out_collection=custom_collection
                    )
                    print("custom_box = Box([1, 2, 3])")
                    print("custom_box.map(double,wrap=custom_wrap,out_collection=custom_collection")
                    print("doubled_custom")
                    print(doubled_custom)  # Output: Box(CustomCollection([2, 4, 6]))
                    print("doubled_custom.to_list()")
                    print(doubled_custom.to_list())
                elif test_name == "Lazy_integers":
                    def lazy_integers():
                        i = 0
                        while True:
                            yield i
                            i += 1

                    lazy_box = Box(lazy_integers())
                    lazy_doubled = lazy_box.map(double)
                    print("lazy_doubled")
                    print(lazy_doubled)  # Output: Box(<lazy>)
                    print("list(islice(lazy_doubled.value, 5))")
                    print(list(islice(lazy_doubled.value, 5)))  # Output: [0, 2, 4, 6, 8]
                    print('Take the first 5')
                    taken, rest=lazy_doubled.take(5)
                    print(taken)
                    print('Take 5 more')
                    taken, rest=lazy_doubled.take(5)
                    print(taken)

                elif test_name == "Generator":
                    def gen():
                        yield from range(1, 4)

                    gen_box = Box[Generator[int, None, None], int](gen())
                    print("gen_box=yield from range(1, 4)")
                    print(gen_box)  # Output: Box(<lazy>)
                    doubled_gen = gen_box.map(double)
                    print("doubled_gen.to_list()")
                    print(doubled_gen.to_list())  # Output: [2, 4, 6]

                elif test_name == "Infinite_stream":
                    stream_box = Box[Iterator[int], int](count(1))
                    print("stream_box")
                    print(stream_box)  # Output: Box(<lazy>)
                    doubled_stream = stream_box.map(double)
                    print(list(islice(doubled_stream, 5)))  # Output: [2, 4, 6, 8, 10]
                    print('Take the first 5')
                    taken, rest=doubled_stream.take(5)
                    print(taken)
                    print('Take 5 more')
                    taken, rest=rest.take(5)
                    print(taken)

                elif test_name == "Lazy_iterator":
                    lazy_box = Box[Iterator[int], int](iter([1, 2, 3]))
                    print("lazy_box=Box[Iterator[int], int](iter([1, 2, 3]))")
                    print(lazy_box)  # Output: Box(<lazy>)
                    doubled_lazy = lazy_box.map(double)
                    print("list(doubled_lazy)=list(lazy_box.map(double))=")
                    print(list(doubled_lazy))  # Output: [2, 4, 6]

                elif test_name == "Custom_iterable":
                    class CustomIterable:
                        def __iter__(self):
                            return iter([1, 2, 3])

                    custom_box = Box[CustomIterable, int](CustomIterable())
                    print("custom_box=iter([1, 2, 3])=")
                    print(custom_box)  # Output: Box([1, 2, 3])
                    doubled_custom = custom_box.map(double)
                    print("doubled_custom.to_list()=custom_box.map(double).to_list()=")  
                    print(doubled_custom.to_list())  # Output: [2, 4, 6]

                elif test_name == "Single_value":
                    single_box = Box[List[int], int]([5])
                    print("single_box=Box[List[int], int]([5])=")
                    print(single_box)  # Output: Box([5])
                    doubled_single = single_box.map(double)
                    print("list(doubled_single)=single_box.map(double)=")
                    print(list(doubled_single))  # Output: [10]

                elif test_name == "Iterable_but_not_iterator":
                    iterable_box = Box([1, 2, 3])
                    print("iterable_box=")
                    print(iterable_box)  # Output: Box([1, 2, 3])
                    print("list(iterable_box.map(double))")
                    print(list(iterable_box.map(double)))  # Output: [2, 4, 6]
                elif test_name == "File_stream":
                    test_file_stream()
                if test_name == "Strict_conversion":
                    # Test with a lazy sequence
                    lazy_box = Box(map(lambda x: x * 2, range(5)))
                    print("lazy_box before strict():")
                    print(lazy_box)
                    lazy_box.strict()
                    print("lazy_box after strict():")
                    print(lazy_box)
                    assert isinstance(lazy_box.value, list), "strict() should convert to a list"
                    assert lazy_box.value == [0, 2, 4, 6, 8], "strict() result incorrect"

                    # Test with a dictionary
                    dict_box = Box(iter({'a': 1, 'b': 2, 'c': 3}.items()))
                    print("dict_box before strict():")
                    print(dict_box)
                    dict_box.strict()
                    print("dict_box after strict():")
                    print(dict_box)
                    assert isinstance(dict_box.value, dict), "strict() should convert to a dict"
                    assert dict_box.value == {'a': 1, 'b': 2, 'c': 3}, "strict() result incorrect for dict"

                elif test_name == "To_dict_conversion":
                    # Test with a list
                    list_box = Box([1, 2, 3])
                    result = list_box.to_dict()
                    print("to_dict() result for list:")
                    print(result)
                    assert result == {0: 1, 1: 2, 2: 3}, "to_dict() incorrect for list"

                    # Test with a dictionary
                    dict_box = Box({'a': 1, 'b': 2, 'c': 3})
                    result = dict_box.to_dict()
                    print("to_dict() result for dict:")
                    print(result)
                    assert result == {'a': 1, 'b': 2, 'c': 3}, "to_dict() incorrect for dict"

                    # Test with a lazy sequence
                    lazy_box = Box(map(lambda x: (chr(97 + x), x + 1), range(3)))
                    result = lazy_box.to_dict()
                    print("to_dict() result for lazy sequence:")
                    print(result)
                    assert result == {'a': 1, 'b': 2, 'c': 3}, "to_dict() incorrect for lazy sequence"

                elif test_name == "Forked_take":
                    # Test with a list
                    list_box = Box([1, 2, 3, 4, 5])
                    taken, rest = list_box.forked_take(3)
                    print("forked_take() result for list:")
                    print(f"Taken: {taken}")
                    rest_list = list(rest)
                    print(f"Rest: {rest_list}")
                    assert taken == [1, 2, 3], "forked_take() taken incorrect"
                    assert rest_list == [4, 5], "forked_take() rest incorrect"

                    # Test with a lazy sequence
                    lazy_box = Box(map(lambda x: x * 2, range(1, 6)))
                    taken, rest = lazy_box.forked_take(3)
                    print("forked_take() result for lazy sequence:")
                    print(f"Taken: {taken}")
                    rest_list = list(rest)
                    print(f"Rest: {rest_list}")
                    assert taken == [2, 4, 6], "forked_take() taken incorrect for lazy sequence"
                    assert rest_list == [8, 10], "forked_take() rest incorrect for lazy sequence"

                    # Test that original box is unchanged
                    print("Original list_box after forked_take:")
                    original_list = list(list_box)
                    print(original_list)
                    assert original_list == [1, 2, 3, 4, 5], "Original box changed after forked_take"


            except Exception as e:
                print(f"Error in {test_name} test: {str(e)}")
                traceback.print_exc()  # This will print the full stack trace

def setup_logging(log_level):
    if log_level == 'NONE':
        logging.disable(logging.CRITICAL)
    elif log_level != 'DEFAULT':
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        logging.basicConfig(level=numeric_level)
    else:
        # 'DEFAULT' case is handled in the main logic
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Box class tests or main program")
    parser.add_argument('--log-level', default='DEFAULT', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NONE', 'DEFAULT'],
                        help='Set the logging level')
    parser.add_argument('--test', action='store_true', help='Run tests')
    return parser.parse_args()

def main():
    # Your main program logic here
    pass

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set default log levels
    if args.log_level == 'DEFAULT':
        if args.test:
            args.log_level = 'DEBUG'
        else:
            args.log_level = 'INFO'
    
    setup_logging(args.log_level)
    
    if args.test:
        logging.info("Running tests with log level: %s", args.log_level)
        test()
    else:
        logging.info("Running main program with log level: %s", args.log_level)
        main()
