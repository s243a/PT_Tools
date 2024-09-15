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
import copy
from collections.abc import MutableSequence, MutableMapping

VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)

logging.Logger.verbose = verbose

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
class SafeListIterator:
    def __init__(self, lst):
        self.list = lst
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.list):
            item = self.list[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration
class LazySequence:
    def __init__(self, iterator, underlying=None, p_box=None):
        self.iterator = iterator
        
        self.exhausted = False
        self.container=p_box
        if underlying is not None and p_box is None:
            warning('LazySequence does not know the container for underlying')
        if underlying is None and p_box is not None:
            warning('LazySequence.underlying is nonem attepting to extract from container')
            underlying=p_box.value
        self.underlying = underlying   


    def __iter__(self):
        return self
    def __next__(self):
        if self.exhausted:
            raise StopIteration        
        try:
            item = next(self.iterator)
            #I don't think we need the following code (suggested by Claude.AI)/
            #It should be taken care of in map
            #if isinstance(self.underlying, list):
            #    self.underlying.append(item)
            #elif isinstance(self.underlying, dict):
            #    if isinstance(item, tuple) and len(item) == 2:
            #        self.underlying[item[0]] = item[1]
            #    else:
            #        self.underlying[len(self.underlying)] = item       
            return item
        except StopIteration:
            #Not sure we want to reset the itter as per Claude.AI's suggestion
            #Also this won't always be possible. 
            logging.debug("Reached the end of the itteration")     
            self.exhausted = True
            if self.container is not None:
                self.container.value=self.underlying
                self.container.transient
            #self.iterator = iter(self.underlying)  # Reset iterator for future iterations
            raise
        except IOError:
            print("An I/O error occurred, in LazySequence.__next__")
            traceback.print_exc()
            raise
        except ValueError:
            print("Invalid data encountered, in LazySequence.__next__")
            traceback.print_exc()
            raise            
        except Exception as e:
            # Log the error or handle it as appropriate
            logging.error("Itteration has stopped Unexpectedly in LazySequence")
            traceback.print_exc()
            raise  # Re-raise the exception

    def __repr__(self):
        return f"LazySequence({self.underlying})"

# I represents the type of the iterable/container
# T represents the type of the elements within the iterable        
class Box(Generic[I, T]):
    def __init__(self, 
                 value: I,
                 wrap: Optional[Callable[[Any, R, Optional[int]], None]] = None,
                 init: Optional[Callable[[], Any]] = None,
                 mutable: bool = False,
                 out_collection = None,
                 transient: bool = False):
        self.value = value
        self.wrap = wrap
        self.mutable = mutable
        self.init = init
        self.out_collection=None
        self.transient = transient 

        if self.mutable and not self._is_safely_mutable(value):
            warnings.warn(
                f"Mutable Box created with potentially unsupported type: {type(value)}. "
                "Behavior for mutable operations is only guaranteed for list-like and dict-like types. "
                "Custom types may require specialized handling.",
                UserWarning
            )

        if self.wrap is not None:
            warnings.warn(
"""Custom wrap function provided. Ensure it correctly handles the underlying data structure
by using assignment rather than append, as append causes infinite loops""",
                UserWarning
            )       
    def _is_safely_mutable(self, value):
        return (isinstance(value, (MutableSequence, MutableMapping)) or 
                self.transient or
                hasattr(value, '__setitem__'))                
    def unBox(self) -> I:
        if self.mutable:
            return self.value
        else:
            raise Exception("Cannot unbox immutable object")

    def __set_wrap__(self, 
        wrap: Optional[Callable[[Any, R, Optional[int]], None]],
        mutable: bool = True,
        out_collection = None) -> Callable[[Any, R, Optional[int]], None]:
        if out_collection is None:
            warnings.warn("please provide out_collection in __Set_wrap__")
            out_collection=self.__set_out_collection__(mutable=mutable)
        test_col = out_collection

        if wrap is not None:
            logger.debug("setting custom wrap")
            return wrap    
        elif self.wrap is not None:
            logger.debug("Using Box's wrap function")
            return self.wrap
        #elif self.mutable and mutable:
        #    #logger.debug("Using default mutable wrap function")
        elif isinstance(self.value, MutableSequence): #More general than List
            if (self.mutable and mutable):
                return lambda coll, item, index: coll.__setitem__(index, item)
            else:
                return lambda coll, item, _: coll.append(item) #if hasattr(coll, 'append') else None            
        elif isinstance(self.value, MutableMapping) or \
           hasattr(self.value, '__setitem__'): #More general than isinstance(self.value, dict):
                return lambda coll, item, key: coll.__setitem__(key, item)
        elif self.mutable and mutable or (out_collection == self.value): #ToDo: check this else branch. Maybe we want to thow an error instead.
            warnings.warn("No __setitem__ method in out collection, using setattr instead")
            return lambda coll, item, index: setattr(coll, f'item_{index}', item)                
        else:
            if hasattr(out_collection, 'append'):
                return lambda coll, item, _: coll.append(item)
            else:
                raise("No suitable function to return in __set_wrap__")


    def __set_out_collection__(self,
        out_collection: Optional[O],
        mutable: bool = True) -> Any:

        if out_collection is not None:
            if callable(out_collection): #ToDo: make sure we want to do this.
                return out_collection()
            return out_collection
        elif self.mutable and mutable:
            return self.value
            #return self.value.__class__()  # Create a new instance of the same type        
        elif self.init is not None:
            return self.init()            
        #elif self.mutable and mutable:
        #    return self.value
        else:
            # Try to create a new instance of the same type
            try:
                new_instance = type(self.value)()
                # Check if we can assign or append to the new instance                   
                #return self.value.class() 
                if hasattr(new_instance, '__setitem__') or hasattr(new_instance, 'append'):
                    return new_instance
            except:
                return []

        # If we can't create a suitable new instance, choose an appropriate default
        if isinstance(self.value, MutableMapping):
            return {}
        elif isinstance(self.value, MutableSequence) or hasattr(self.value, '__iter__'):
            return []
        else:
            # For types we don't recognize, we'll use a list as a safe default
            return [] 
    def map(self, 
            f: Callable[[T], R], 
            wrap: Optional[Callable[[Any, R, Optional[int]], None]] = None,
            out_collection: Optional[O] = None,
            mutable: bool = True
    ) -> 'Box[O, R]':
        out_collection = self.__set_out_collection__(out_collection, mutable=mutable) 
        wrap_func = self.__set_wrap__(wrap,mutable=mutable,out_collection=out_collection)
        
        logging.debug(f'wrap={lamtest(wrap_func)}')
        #print("wrap="+inspect.getsource(wrap))
        #print("out_collection="+str(out_collection))
        def lazy_map() -> Iterator[R]:
            if isinstance(self.value, MutableMapping) or \
               hasattr(self.value, 'items'): #More general than isinstance(self.value, dict):
                for k, v in self.value.items():
                    result = f(v)
                    logger.debug(f"Mapping: key={k}, original value={v}, mapped value={result}")                    
                    #if self.mutable and mutable and out_collection is self.value:
                    #    #wrap(out_collection, result[1], result[0])
                    #    wrap(out_collection, result, k)
                    wrap_func(out_collection, result, k)
                    yield k, result
            elif isinstance(self.value, MutableSequence) or \
               hasattr(self.value, '__setitem__'):
                for i, item in enumerate(SafeListIterator(self.value)):
                    result = f(item)
                    wrap_func(out_collection, result, i)
                    yield result
            else:  # Handle generators and other iterables
                if out_collection == self.value:
                    raise ValueError("cannot safely itterate over self.vaue")
                for i, item in enumerate(self.value):
                    result = f(item)
                    wrap_func(out_collection, result, i)
                    yield result
        mutable_out=mutable and self.mutable
        iterator = lazy_map()
        return Box(LazySequence(iterator, out_collection, self),
                        wrap=wrap, init=self.init, mutable=mutable_out, transient=True)
        #if self.mutable and mutable and out_collection is self.value:
        #    # For mutable in-place modification, we return a Box with the same value
        #    # but wrap it in a LazySequence to preserve lazy evaluation
        #    logger.debug("Default lazy_map for mutable objects")
        #    return Box(LazySequence(iterator, out_collection), 
        #               wrap=wrap, init=self.init, mutable=True)
        #else:
        #    # For immutable or new collections, we return a Box with the iterator
        #    logger.debug("Default lazy_map for immutable objects")
        #    return Box(LazySequence(iterator, out_collection),
        #               wrap=wrap, init=self.init, mutable=mutable)

    def __iter__(self):
        if isinstance(self.value, LazySequence):
            logger.debug("itterating box over LazySequence")
            yield from self.value
            #for item in self.value:
            #    yield item
            if isinstance(self.value, LazySequence):
                logging.debug("Setting self.value to underlying within box")
                self.value = self.value.underlying
                self.transient = False  
        elif isinstance(self.value, (Iterator, map)):
            yield from self.value
        elif hasattr(self.value, '__iter__'):
            yield from self.value
        elif hasattr(self.value, 'data') and hasattr(self.value.data, '__iter__'):
            yield from self.value.data

        else:
            traceback.print_exc()  # This will print the full stack trace
            raise TypeError(f"Cannot iterate over {type(self.value)}")        
        #if isinstance(self.value, itertools.chain):
        #    self.value = list(self.value)
        #return iter(self.value)

    def __repr__(self) -> str:
        if isinstance(self.value, (Iterator, map)):
            return f"Box(<lazy>)"
        return f"Box({self.value})"


    def take(self, N: int = 0) -> List[T]:
        """
        Return a list containing the first N items from the collection (or all items if N=0).
        This method advances the internal iterator.
        
        If N is 0 or larger than the collection size, all remaining items will be returned.
        
        Raises:
            ValueError: If called on a non-mutable Box. Use forked_take for immutable operations.
        """
        if not (self.mutable or self.transient):
            raise ValueError("take() cannot be used on immutable, non-transient Box objects. Use forked_take() instead.")

        if isinstance(self.value, LazySequence):
            if N == 0:
                return list(self.value.underlying)
            return [next(self.value) for _ in range(N)]

        self.value=iter(self.value)
        if N == 0:
            taken = list(self.value)
        else:
            #Not sure if the following works on infite streams
            #taken = list(islice(self.value, N))
            #So Claude.AI proposed the following:
            for _ in range(N):
                try:
                    taken.append(next(self.value))
                except StopIteration:
                    break

        #return taken, Box(iterator, wrap=self.wrap, init=self.init, mutable=self.mutable)
        return taken
    #@experimental
    def forked_take(self, N: int) -> Tuple[List[T], 'Box[Union[Iterator[T], I], T]']:
        """
        EXPERIMENTAL: This method is for future development and may change.
        Takes N items from the stream, returning them as a list along with a new Box
        containing the rest of the stream.
        This creates a fork in the stream, allowing for Haskell-like list operations.

        This method is safe for both mutable and immutable Box objects.

        For Iterator and map objects, this method modifies the original Box.
        For other types, the original Box remains unchanged.

        If N exceeds the number of items, all available items are returned in the first list,
        and the second Box will contain an empty iterator.
        """
        if isinstance(self.value, (Iterator, map)):
            iter1, iter2 = itertools.tee(iter(self.value))
            self.value=iter1
        else:
            iter2 = iter(self.value)
        taken = list(itertools.islice(iter2, N))
        return taken, Box(iter2, wrap=self.wrap, init=self.init, mutable=self.mutable)
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
        self.strict()
        try:
            return list(iter(self.value))
        except TypeError:
            raise TypeError(f"Cannot convert to list: {type(self.value)} object is not iterable")
    def to_dict(self):
        # If the value is already a dictionary, return it directly
        if isinstance(self.value, dict):
            if self.mutable:
                return self.value
            else:
                return copy.deepcopy(self.value)
        # If the value is a LazySequence, we need to evaluate it
        if isinstance(self.value, LazySequence):
            self.strict() 

        # Now check again if it's a dictionary after potential strictification
        if isinstance(self.value, dict):
            return self.value

        # For other types, create a dictionary
        out = {}
        if isinstance(self.value, (list, tuple)):
            # If it's a list or tuple, use enumerate for indices
            for i, item in enumerate(self.value):
                out[i] = item
        else:
            # For other iterable types, try to iterate and use indices
            try:
                for i, item in enumerate(self.value):
                    if isinstance(item, tuple) and len(item) == 2:
                        # If the item is a key-value pair, use it directly
                        out[item[0]] = item[1]
                    else:
                        out[i] = item
            except TypeError:
                # If it's not iterable, just return a single-item dictionary
                out[0] = self.value

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

    

    def strict(self, verbose=True,log_concrete=False):
        """
        Iterate through the entire collection, making it concrete.
        If verbose is True, print each item.
        """
        logging.debug(f"self.value is a {type(self.value)}")
        if isinstance(self.value, (LazySequence)):
            for item in self.value:
                logger.debug(item)
                logger.debug(f"self.value.underlying={self.value.underlying}")
            self.value=self.value.underlying            
        elif isinstance(self.value, (Iterator, map)):
            iter1, iter2 = itertools.tee(iter(self.value))
            concrete_value, item_type, iter2 = self.__infer_type_and_create_concrete_value(iter2)

            try:
                for item in iter2:               
                    self.__add_to_concrete_value(concrete_value, item, item_type)
                    if verbose:                    
                        if isinstance(item, tuple) and len(item) == 2:
                            key, value = item
                            logger.debug(f"Added to concrete value: key={key}, value={value}")
                        else:
                            logger.debug(f"Added to concrete value: {item}")

                self.value = concrete_value
            except Exception as e:
                logger.error(f"Error during Box.strict() evaluation: {e}")
                logger.debug(f"Last processed item: {item}")
                self.value=iter1
            finally:
                # Ensure we clean up our references to the teed iterators
                del iter1
            del iter2            
        elif verbose and log_concrete: #The orginaly idea here was itterating though the object should make it stract as a side effect
                      #I'm not sure this is the case.
            # For non-iterator types, we just log the items without changing self.value  
            logger.debug("Value is already concrete, logging contents:")        
            for item in self.value:
                logger.debug(item)
        return self
        if isinstance(self.value, (Iterator, map, LazySequence)):
            itt=iter(self.value)
            iter1, iter2 = itertools.tee(self.value)
            concrete_value, item_type, iter2 = self.__infer_type_and_create_concrete_value(iter2)
            try:
                for item in iter2:               
                    self.__add_to_concrete_value(concrete_value, item, item_type)
                    if verbose:                    
                        if isinstance(item, tuple) and len(item) == 2:
                            key, value = item
                            logger.debug(f"Added to concrete value: key={key}, value={value}")
                        else:
                            logger.debug(f"Added to concrete value: {item}")

                self.value = concrete_value
                iter1=None
                iter2=None
            except Exception as e:
                logger.error(f"Error during Box.strict() evaluation: {e}")
                logger.debug(f"Last processed item: {item}")
                self.value=iter1
        elif verbose: #The orginaly idea here was itterating though the object should make it stract as a side effect
                      #I'm not sure this is the case.
            for item in self.value:
                logger.debug(item)
        return self

    def __infer_type_and_create_concrete_value(self,itt):
        if self.init is not None:
            return self.init(), None, itt
        
        # Try to peek at the first item without consuming the iterator
        try:
            first_item = next(itt)
            # Put the item back into an iterator
            itt=itertools.chain([first_item], itt)
        except StopIteration:
            # Iterator is empty
            return [], None, itt

        if isinstance(first_item, tuple) and len(first_item) == 2:
            # Likely a key-value pair, assume it's a dictionary
            logger.debug(f"Inferred type: {type(first_item)}")
            return {}, dict, itt
        elif isinstance(first_item, Iterable) and not isinstance(first_item, str):
            # It's some other kind of iterable, assume it's a list of lists
            logger.debug(f"Inferred type: {type(first_item)}")
            return [], list, itt
        else:
            # Default to a list
            logger.debug(f"Inferred type: {type(first_item)}")
            return [], list, itt

    def __add_to_concrete_value(self, concrete_value, item, item_type):
        if self.wrap is not None:
            logger.debug("using custom wrap in __add_to_concrete_value")
            if item_type == dict:
                logger.debug(f"Adding to concrete value. Type: {item_type}")
                
                key, value = item
                logger.debug(f"Adding to dict: key={key}, value={value}")
                self.wrap(concrete_value, value, key)
            else:
                self.wrap(concrete_value, item, len(concrete_value))
        elif item_type == dict:
            logger.debug(f"Unpacking({str(item)} in __add_to_concrete_value")
            key, value = item
            concrete_value[key] = value
        elif isinstance(concrete_value, list):
            concrete_value.append(item)
        else:
            setattr(concrete_value, f'item_{len(concrete_value)}', item)  
    def peek(self, n: int = 1) -> List[T]:
        """
        Look at the next n items without consuming them.
        """
        if isinstance(self.value, list):
            return self.value[:n]
        
        iterator = iter(self.value)
        #Todo: Consider using Tee here instead, like we do with forked_take
        peeked = list(itertools.islice(iterator, n))
        self.value = itertools.chain(peeked, iterator)
        return peeked

    def take_while(self, predicate: Callable[[T], bool]) -> 'Box[List[T], T]':
        """
        Take elements as long as they satisfy the given predicate.
        """
        taken = list(itertools.takewhile(predicate, self.value))
        return Box(taken, wrap=self.wrap, init=self.init, mutable=self.mutable)

    def split_at(self, index: int) -> Tuple['Box[List[T], T]', 'Box[List[T], T]']:
        """
        Split the Box at the given index, returning two new Boxes.
        """
        #Todo: consider using tee here instead.
        iterator = iter(self.value)
        first = list(itertools.islice(iterator, index))
        rest = list(iterator)
        return (Box(first, wrap=self.wrap, init=self.init, mutable=self.mutable),
                Box(rest, wrap=self.wrap, init=self.init, mutable=self.mutable))

#    def zip_with(self, other: 'Box[Any, Any]', func: Callable[[T, Any], R]) -> 'Box[List[R], R]':
#        """
#        Combine two Boxes element-wise using the given function.
#        """
#        zipped = list(map(func, self.value, other.value))
#        return Box(zipped, wrap=self.wrap, init=self.init, mutable=self.mutable)
    def zip_with(self, *others: 'Box', func: Union[Callable[..., Any], None] = None) -> 'Box':
        """
        Combine this Box with other Boxes element-wise using the given function.
        If no function is provided, it returns a Box of tuples (like built-in zip).
        
        :param others: Other Box objects to zip with
        :param func: Function to apply to zipped elements. If None, returns tuples.
        :return: A new Box with the zipped and processed elements
        """
        if func is None:
            # If no function is provided, use tuple constructor
            func = lambda *args: args
        
        iterables = [iter(self.value)] + [iter(other.value) for other in others]
        
        def zip_generator():
            while True:
                try:
                    values = [next(it) for it in iterables]
                    yield func(*values)
                except StopIteration:
                    return  # End the generator when any iterator is exhausted

        return Box(zip_generator(), wrap=self.wrap, init=self.init, mutable=self.mutable)
#################################### Test the function ########################
def double(x: int) -> int:
    return x * 2
def test_strict_with_custom_wrap():
    def custom_wrap(collection, item, index):
        print(f"Custom wrap called with index: {index}, item: {item}")
        collection.append(item * 2)

    lazy_sequence = map(lambda x: x + 1, range(5))
    box = Box(lazy_sequence, wrap=custom_wrap, mutable=True)
    
    logger.info(f"""
box = 
    Box(lazy_sequence=map(lambda x: x + 1, range(5)),
    wrap=collection.append(item * 2), 
    Mutable=True)""")
    logger.debug(f"Box before strict(): {box}")
    
    box.strict()
    
    logger.info(f"Box after strict(): {box}")

    
    assert box.value == [2, 4, 6, 8, 10], "strict() didn't apply custom wrap correctly"
    print("strict() with custom wrap test passed")
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

        # This should show Box(<lazy>)
        logger.info(f"file_box (before operations): {file_box}") 
        logger.info("File contents: " + r"1\n2\n3\n4\n5\n")

        # Map operation to double each number
        doubled_file = file_box.map(double)

        # This should also show Box(<lazy>)
        logger.debug(f"doubled_file: {doubled_file}")

        result = list(doubled_file)
        # This should print [2, 4, 6, 8, 10]
        logger.debug(f"List(doubled_file): {result}")
        assert result == [2, 4, 6, 8, 10]

        # Create a new Box for testing take() method
        new_file_box = Box(file_reader(temp_file_name))
        new_doubled_file = new_file_box.map(double)

        # Test taking first 3 items
        logger.info("Now test the take() method w/ Box(<file stream>)")
        logger.debug("Box(file_reader(1\\n2\\n3\\n4\\n5\\n))")
        taken, rest = new_doubled_file.forked_take(3)
        # This should print [2, 4, 6]
        logger.info(f"First 3 items: {taken}")
        assert taken == [2, 4, 6]

        # Test remaining items
        remaining = list(rest)

        # This should print [8, 10]
        logger.info(f"Remaining items: {remaining}")
        assert remaining == [8, 10]

    finally:
        # Clean up: remove the temporary file
        os.unlink(temp_file_name)

def test():

    # Example usage

    tests={
     "Immutable_list" : False,
     "Mutable_list" : False,
     "Dictionary" : False,
     "Dictionary_with_custom_wrap" : False,
     "Custom_collection" : False,
     "Lazy_integers" : False,
     "Generator" : False,
     "Infinite_stream" : False,
     "Lazy_iterator" : False, 
     "Custom_iterable" : False,
     "Single_value" : True,
     "Iterable_but_not_iterator" : False,
     "File_stream" : False,
     "Strict_conversion": False,
     "To_dict_conversion": False,
     "Forked_take": False,
     "Strict_with_custom_wrap" : False,
     "Test_peek" : False,
     "Split_at" : False,
     "Zip_with" : False
    }
    
    args = parse_arguments()
    if test_all:
        print("enabling tests")
        for k in tests:

            tests[k]=True
    for test_name, enabled in tests.items():
        if enabled:
            print(f"\n# Running test: {test_name}")
            try:   
                if test_name == "Immutable_list":
                    logging.info("immutable_box = Box([1, 2, 3], mutable=False)")
                    immutable_box = Box([1, 2, 3], mutable=False)
                    logger.debug(f"immutable_box={immutable_box}")
                    doubled_immutable = immutable_box.map(double)
                    logger.debug(f"doubled_immutable={doubled_immutable}")
                    doubled_immutable_to_list = doubled_immutable.to_list()
                    logging.info(f"doubled_immutable.to_list()={doubled_immutable_to_list}")

                    assert doubled_immutable_to_list == [2, 4, 6], "Unexpected result for immutable list"

                elif test_name == "Mutable_list":
                    mutable_box = Box([1, 2, 3], mutable=True)
                    logging.info("mutable_box = Box([1, 2, 3], mutable=True)")
                    logger.debug(f"mutable_box.to_list()={mutable_box.to_list()}")
                    doubled_mutable = mutable_box.map(double)
                    logger.debug(f"doubled_mutable={doubled_mutable.to_list()}")# Output: Box([2, 4, 6])
                    logger.debug(f"mutable_box.unBox()={mutable_box.unBox()}")  # Output: [2, 4, 6]
                    logging.info(f"mutable_box.to_list() == ={mutable_box.to_list()}")  # Output: [2, 4, 6]
                    assert mutable_box.to_list() == [2, 4, 6]

                if test_name == "Dictionary":
                    dict_box = Box({'a': 1, 'b': 2, 'c': 3}, mutable=True)

                    logging.info("Original dict_box: Box({'a': 1, 'b': 2, 'c': 3}, mutable=True)")
                    logger.debug(dict_box.unBox())
                    
                    doubled_dict = dict_box.map(double)
                    logger.debug("doubled_dict (before iteration):")
                    logger.debug(doubled_dict)
                    
                    logger.debug("Strict doubled_dict:")
                    doubled_dict.strict(verbose=True)
                    logger.debug(doubled_dict.to_dict())

                    logger.debug("dict_box after mapping #1: ")
                    logger.debug(dict_box.unBox())
                    logger.debug("dict_box after mapping #2:")
                    logger.debug(dict_box.to_dict())                   
                    logger.debug("Iterating over doubled_dict:")
                    for k, v in doubled_dict.unBox().items():
                        logger.debug(f"{k}: {v}")

                    logging.info("Final state of dict_box:")
                    logging.info(dict_box.unBox())
                    assert dict_box.unBox() == {'a': 2, 'b': 4, 'c': 6}, "dict_box not correctly updated"
                    logger.debug("Final state of doubled_dict #1:")
                    logger.debug(doubled_dict.unBox())
                    logging.info("Final state of doubled_dict #2:")
                    logging.info(doubled_dict.to_dict())
                    assert doubled_dict.to_dict() == {'a': 2, 'b': 4, 'c': 6}, "doubled_dict incorrect"

                    print("Dictionary test passed")
                elif test_name == "To_dict_immutability":
                    # Test with a mutable dictionary Box
                    mutable_dict_box = Box({'a': 1, 'b': 2, 'c': 3}, mutable=True)
                    logging.info("mutable_dict_box = Box({'a': 1, 'b': 2, 'c': 3}, mutable=True)")
                    mutable_result = mutable_dict_box.to_dict()
                    mutable_result['a'] = 10  # Modify the result
                    logging.info(f"mutable_result['a'] = {mutable_result['a']}= {mutable_dict_box.unBox()['a']} = mutable_dict_box.unBox()['a']")
                    assert mutable_dict_box.unBox()['a'] == 10, "Mutable dict_box not updated"

                    # Test with an immutable dictionary Box
                    immutable_dict_box = Box({'a': 1, 'b': 2, 'c': 3}, mutable=False)
                    immutable_result = immutable_dict_box.to_dict()
                    immutable_result['a'] = 10  # Modify the result
                    logging.info(f"mutable_result['a'] = {mutable_result['a']}!= {mutable_dict_box.unBox()['a']} = mutable_dict_box.unBox()['a']")
                    assert immutable_dict_box.unBox()['a'] == 1, "Immutable dict_box incorrectly updated"

                    logging.info("To_dict immutability test passed")
                elif test_name == "Dictionary_with_custom_wrap":
                    def custom_dict_wrap(d, item, key):
                        print(f"Custom wrap called with key: {key}, value: {item}")
                        d[key] = item

                    dict_box = Box({'a': 1, 'b': 2, 'c': 3}, 
                                   wrap=custom_dict_wrap,
                                   init=dict,
                                   mutable=True)
                    logging.info("Original dict_box:")
                    logging.info(dict_box.unBox())
                    
                    doubled_dict = dict_box.map(double)
                    logger.debug("doubled_dict (before iteration):")
                    logger.debug(doubled_dict)
                    
                    logger.debug("dict_box after mapping (before iteration):")
                    logger.debug(dict_box.unBox())
                    
                    logger.debug("Iterating over doubled_dict:")
                    list(doubled_dict)  # Force iteration

                    logging.info("Final state of dict_box:")
                    logging.info(dict_box.unBox())               
                    assert dict_box.unBox() == {'a': 2, 'b': 4, 'c': 6}, "doubled_dict incorrect"

                elif test_name == "Custom_collection":
                    class CustomCollection:
                        def __init__(self):
                            self.data = []
                        def add(self, item):
                            self.data.append(item)
                        def __iter__(self):
                            return iter(self.data)    
                        def __repr__(self):
                            return f"CustomCollection({self.data})"
                        def __len__(self):
                            return len(self.data)                        

                    custom_box = Box([1, 2, 3])
                    custom_collection = CustomCollection()
                    def custom_wrap(coll, item, _):
                        coll.add(item)
                    doubled_custom = custom_box.map(
                        double, 
                        wrap=custom_wrap, 
                        out_collection=custom_collection
                    )
                    logging.info("custom_box = Box([1, 2, 3])")
                    logger.debug("custom_box.map(double,wrap=custom_wrap,out_collection=custom_collection")
                    logger.debug("doubled_custom")
                    logger.debug(doubled_custom)  # Output: Box(CustomCollection([2, 4, 6]))
                    logging.info("doubled_custom.to_list()")
                    logging.info(doubled_custom.to_list())
                    assert doubled_custom.to_list() == [2, 4, 6], "doubled_dict incorrect"
                elif test_name == "Lazy_integers":
                    def lazy_integers():
                        i = 0
                        while True:
                            yield i
                            i += 1

                    lazy_box = Box(lazy_integers())
                    logging.info(f"lazy_box = Box(lazy_integers())={lazy_box}")
                    lazy_doubled = lazy_box.map(double)
                    print("lazy_doubled")
                    print(lazy_doubled)  # Output: Box(<lazy>)
                    logger.debug('Take the second 5, via list(islice(lazy_doubled.value, 5))')
                    taken=list(islice(lazy_doubled.value, 5)) # Output: [0, 2, 4, 6, 8]
                    logger.debug(f"taken={taken}")
                    assert taken == [0, 2, 4, 6, 8], "Doubled Lazy integrs not correct for 1st 5 taken"
                    logger.debug('Take the second 5, via lazy_doubled.take(5)')
                    taken=lazy_doubled.take(5) # Output: [10, 12, 14, 16, 18]
                    logger.debug(f"taken={taken}")
                    assert taken == [10, 12, 14, 16, 18], "Doubled Lazy integrs not correct for 2rd 5 taken"
                    logging.info('Take the third 5, via lazy_doubled.take(5)')
                    taken, rest=lazy_doubled.forked_take(5) # Output: [20, 22, 24, 26, 28]
                    logging.info(f"taken={taken}")
                    assert taken == [20, 22, 24, 26, 28], "Doubled Lazy integrs not correct for 3rd 5 taken"

                elif test_name == "Generator":
                    def gen():
                        yield from range(1, 4)

                    gen_box = Box[Generator[int, None, None], int](gen())
                    logging.info("gen_box=yield from range(1, 4)")
                    logger.debug(gen_box)  # Output: Box(<lazy>)
                    doubled_gen = gen_box.map(double)
                    logging.info("doubled_gen.to_list()")
                    dbl_gen_list=doubled_gen.to_list()
                    logging.info(dbl_gen_list)  # Output: [2, 4, 6]
                    assert dbl_gen_list == [2, 4, 6]

                elif test_name == "Infinite_stream":
                    stream_box = Box[Iterator[int], int](count(1))
                    logging.info("stream_box=Box[Iterator[int], int](count(1))")
                    logger.debug(stream_box)  # Output: Box(<lazy>)
                    doubled_stream = stream_box.map(double)
                    logging.debug("Take the 1rd set of 5 doubled (count seqeunce) - via list(islice(doubled_stream, 5))")
                    taken=list(islice(doubled_stream, 5))
                    logging.debug(f"taken={taken}")  # Output: [2, 4, 6, 8, 10]

                    logging.debug("Take the 2rd set of 5 doubled (count seqeunce) - via doubled_stream.take(5)")
                    taken=doubled_stream.take(5) # Output: [12, 14, 16, 18, 20]
                    logging.debug(f"taken={taken}")
                    assert taken == [12, 14, 16, 18, 20]
                    
                    logging.info("Take the 3rd set of 5 doubled (count seqeunce) - via doubled_stream.take(5)")
                    taken, rest=doubled_stream.forked_take(5) #[22, 24, 26, 28, 30]
                    logging.info(f"taken={taken}")
                    assert taken == [22, 24, 26, 28, 30]

                elif test_name == "Lazy_iterator":
                    lazy_box = Box[Iterator[int], int](iter([1, 2, 3]))
                    logging.info("lazy_box=Box[Iterator[int], int](iter([1, 2, 3]))")
                    logging.debug(lazy_box)  # Output: Box(<lazy>)
                    doubled_lazy = lazy_box.map(double)
                    logging.debug("list(doubled_lazy)=list(lazy_box.map(double))=")
                    lst_doubled_lazy=list(doubled_lazy)  # Output: [2, 4, 6]
                    logging.info(lst_doubled_lazy) 
                    assert lst_doubled_lazy == [2, 4, 6]
                elif test_name == "Custom_iterable":
                    class CustomIterable:
                        def __init__(self):
                            self.data = [1, 2, 3]

                        def __iter__(self):
                            return iter(self.data)

                    custom_iterable = CustomIterable()
                    custom_box = Box[CustomIterable, int](custom_iterable)
                    logger.info(f"custom_box=CustomIterable()={custom_box}")
                    logging.debug(custom_box)  # Output: Box([1, 2, 3])
                    doubled_custom = custom_box.map(double)      
                    logging.info("doubled_custom.to_list()=custom_box.map(double).to_list()=") 
                    lst_doubled_custom=doubled_custom.to_list()
                    logging.info(lst_doubled_custom)  # Output: [2, 4, 6]
                    assert lst_doubled_custom == [2, 4, 6]

                elif test_name == "Single_value":
                    single_box = Box[List[int], int]([5])
                    logging.info("single_box=Box[List[int], int]([5])")
                    logging.debug(f"single_box={single_box}")  # Output: Box([5])
                    doubled_single = single_box.map(double)
                    logging.debug(f"single_box.map(double)={doubled_single}")
                    lst_doubled_single=list(doubled_single)
                    logging.info(f"list(doubled_single)={lst_doubled_single}")  # Output: [10]
                    assert lst_doubled_single == [10]

                elif test_name == "Iterable_but_not_iterator":
                    iterable_box = Box([1, 2, 3])
                    logging.info("Assign: iterable_box = Box([1, 2, 3])")
                    logging.debug(f"ToString: iterable_box = {iterable_box}")  # Output: Box([1, 2, 3])

                    dbl_iterable=iterable_box.map(double)
                    lst_dbl_iterable=list(dbl_iterable)                    
                    logging.debug(f"iterable_box.map(double)={dbl_iterable}")                    
                    logging.info(f"List(iterable_box.map(double))={lst_dbl_iterable}")  # Output: [2, 4, 6]
                    assert lst_dbl_iterable == [2, 4, 6]
                elif test_name == "File_stream":
                    test_file_stream()
                elif test_name == "Strict_conversion":
                    # Test with a lazy sequence
                    lazy_box = Box(map(lambda x: x * 2, range(5)))
                    logging.info("Assignment: lazy_box = Box(map(lambda x: x * 2, range(5)))")
                    logging.debug("lazy_box before strict(): {lazy_box}")
                    lazy_box.strict()
                    logging.info(f"lazy_box after strict(): {lazy_box}")

                    assert isinstance(lazy_box.value, list), "strict() should convert to a list"
                    assert lazy_box.value == [0, 2, 4, 6, 8], "strict() result incorrect"

                    # Test with a dictionary
                    dict_box = Box(iter({'a': 1, 'b': 2, 'c': 3}.items()))
                    logging.debug(f"dict_box before strict(): {dict_box}")

                    dict_box.strict()
                    logging.info(f"dict_box after strict(): {dict_box}")
                    assert isinstance(dict_box.value, dict), "strict() should convert to a dict"
                    assert dict_box.value == {'a': 1, 'b': 2, 'c': 3}, "strict() result incorrect for dict"

                elif test_name == "To_dict_conversion":
                    # Test with a list
                    list_box = Box([1, 2, 3])
                    result = list_box.to_dict()
                    logging.info(f"to_dict() result for list: {result}")
                    assert result == {0: 1, 1: 2, 2: 3}, "to_dict() incorrect for list"

                    # Test with a dictionary
                    dict_box = Box({'a': 1, 'b': 2, 'c': 3})
                    result = dict_box.to_dict()
                    logging.info(f"to_dict() result for dict: {result}")
                    assert result == {'a': 1, 'b': 2, 'c': 3}, "to_dict() incorrect for dict"

                    # Test with a lazy sequence
                    lazy_box = Box(map(lambda x: (chr(97 + x), x + 1), range(3)))
                    result = lazy_box.to_dict()
                    logging.info("to_dict() result for lazy sequence: {result}")
                    assert result == {'a': 1, 'b': 2, 'c': 3}, "to_dict() incorrect for lazy sequence"

                    # Test with a lazy sequence of key-value pairs
                    lazy_box = Box(map(lambda x: (chr(97 + x), x + 1), range(3)))
                    result = lazy_box.to_dict()
                    logging.info("to_dict() result for lazy sequence of key-value pairs:")
                    logging.info(result)
                    assert result == {'a': 1, 'b': 2, 'c': 3}, "to_dict() incorrect for lazy sequence of key-value pairs"

                    # Test with a lazy sequence of non-pairs
                    lazy_box_non_pairs = Box(map(lambda x: x * 2, range(3)))
                    result = lazy_box_non_pairs.to_dict()
                    logging.info("to_dict() result for lazy sequence of non-pairs:")
                    logging.info(result)
                    assert result == {0: 0, 1: 2, 2: 4}, "to_dict() incorrect for lazy sequence of non-pairs"

                    logging.info("All to_dict() tests passed")                    

                elif test_name == "Forked_take":
                    # Test with a list
                    list_box = Box([1, 2, 3, 4, 5])
                    logging.info("list_box = Box([1, 2, 3, 4, 5])")
                    taken, rest = list_box.forked_take(3)
                    rest_list = list(rest)
                    logging.debug("forked_take() result for list:")
                    logging.info(f"Taken: {taken}, Rest: {rest_list}")                   
                    assert taken == [1, 2, 3], "forked_take() taken incorrect"
                    assert rest_list == [4, 5], "forked_take() rest incorrect"
                    print()
                    # Test with a lazy sequence
                    logging.info("lazy_box = Box(map(lambda x: x * 2, range(1, 6)))")
                    lazy_box = Box(map(lambda x: x * 2, range(1, 6)))
                    taken, rest = lazy_box.forked_take(3)
                    rest_list = list(rest)
                    logging.debug("forked_take() result for lazy sequence:")
                    logging.info(f"Taken: {taken}, Rest: {rest_list}")
                    assert taken == [2, 4, 6], "forked_take() taken incorrect for lazy sequence"
                    assert rest_list == [8, 10], "forked_take() rest incorrect for lazy sequence"
                    print()
                    # Test that original box is unchanged           
                    original_list = list(list_box)
                    logging.info("Original list_box after forked_take:")
                    logging.info(f"original_list: {original_list}")
                    assert original_list == [1, 2, 3, 4, 5], "Original box changed after forked_take"
                elif test_name == "Strict_with_custom_wrap":
                    test_strict_with_custom_wrap()
                elif test_name == "Test_peek": 
                    logging.info("Box([1, 2, 3, 4, 5])")              
                    box = Box([1, 2, 3, 4, 5])
                    
                    peeked = box.peek(3)
                    box_to_list=box.to_list()
                    logging.info(f"Peeked: {peeked}, box.to_list(): {box_to_list}")
                    assert peeked == [1, 2, 3], "peek() incorrect"
                    assert box_to_list == [1, 2, 3, 4, 5]
                    box_list = list(box)
                    logging.debug(f"box (list): {box_list}")
                    assert box_list == [1, 2, 3, 4, 5], "peek() changed the box contents"
                    print()
                    logging.debug("Test peek again to ensure it doesn't consume items")
                    peeked_again = box.peek(2)
                    logging.debug(f"Peeked again: {peeked_again}")
                    assert peeked_again == [1, 2], "peek() incorrect on second call"
                    assert list(box) == [1, 2, 3, 4, 5], "peek() changed the box contents after second call"
                    
                    print("peek() test passed")
                elif test_name == "Split_at":  
                    # Test split_at
                    box = Box([1, 2, 3, 4, 5])
                    logging.info("Box([1, 2, 3, 4, 5])")
                    
                    # Test normal split
                    first, rest = box.split_at(3)
                    logging.info(f"Split at 3: {list(first)}, {list(rest)}")
                    assert list(first) == [1, 2, 3] and list(rest) == [4, 5], "split_at(3) incorrect"
                    print()
                    # Test split at start
                    first, rest = box.split_at(0)
                    logging.debug(f"Split at 0: {list(first)}, {list(rest)}")
                    assert list(first) == [] and list(rest) == [1, 2, 3, 4, 5], "split_at(0) incorrect"

                    # Test split at end
                    first, rest = box.split_at(5)
                    logging.debug(f"Split at 5: {list(first)}, {list(rest)}")
                    assert list(first) == [1, 2, 3, 4, 5] and list(rest) == [], "split_at(5) incorrect"

                    # Test split beyond end
                    first, rest = box.split_at(10)
                    logging.debug(f"Split at 10: {list(first)}, {list(rest)}")
                    assert list(first) == [1, 2, 3, 4, 5] and list(rest) == [], "split_at(10) incorrect"

                    print("All split_at tests passed")
                elif test_name == "Zip_with":  
                    box1 = Box([1, 2, 3])
                    box2 = Box([4, 5, 6])
                    box3 = Box([7, 8, 9])

                    # Test with custom function
                    logging.debug("box1 = Box([1, 2, 3])")
                    logging.debug("box1.zip_with(box2=Box([4, 5, 6]), func=lambda x, y: x + y)")
                    zipped = box1.zip_with(box2, func=lambda x, y: x + y)
                    peeked = zipped.peek(3)

                    logging.info(f"Zipped and added (peeked): {peeked}")
                    print()
                    assert peeked == [5, 7, 9], "zip_with() with custom function incorrect"

                    # Test without function (should behave like normal zip)
                    logging.info("zipped_default=box1.zip_with(box2) #Zips w/o function, like normal zip")
                    zipped_default = box1.zip_with(box2)
                    peeked_default = zipped_default.peek(3)
                    logging.info(f"Zipped with default (peeked): {peeked_default}=zipped_default")
                    assert peeked_default == [(1, 4), (2, 5), (3, 6)], "zip_with() default behavior incorrect"
                    print()
                    # Test with more than two boxes
                    zipped_multi = box1.zip_with(box2, box3, func=lambda x, y, z: x + y + z)
                    logging.debug("box3 = Box([7, 8, 9])")
                    logging.debug("box1.zip_with(box2, box3, func=lambda x, y, z: x + y + z)")
                    peeked_multi = zipped_multi.peek(3)
                    logging.info(f"Zipped multi (peeked): {peeked_multi}")
                    assert peeked_multi == [12, 15, 18], "zip_with() with multiple boxes incorrect"

                    # Test with boxes of different lengths
                    short_box = Box([1, 2])
                    zipped_diff_len = box1.zip_with(short_box)
                    peeked_diff_len = zipped_diff_len.peek(3)
                    logging.info(f"Zipped different lengths (peeked): {peeked_diff_len}")
                    assert peeked_diff_len == [(1, 1), (2, 2)], "zip_with() with different length boxes incorrect"

                    print("All zip_with tests passed")
            except Exception as e:
                print(f"Error in {test_name} test: {str(e)}")
                traceback.print_exc()  # This will print the full stack trace

            except Exception as e:
                print(f"Error in {test_name} test: {str(e)}")
                traceback.print_exc()  # This will print the full stack trace
    print("All enabled Box method tests passed")
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
                        choices=["VERBOSE", 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NONE', 'DEFAULT'],
                        help='Set the logging level')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    return parser.parse_args()

def main():
    # Your main program logic here
    pass

if __name__ == "__main__":
    args = parse_arguments()
    test_all=False
    # Set default log levels
    if args.log_level == 'DEFAULT':
        if args.test:
            args.log_level = 'DEBUG'
        elif args.test_all:
            print("Running all tests")
            test_all=True
            #logging.getLogger().setLevel(logging.INFO)
            args.log_level = 'INFO'
        else:
            args.log_level = 'INFO'
    
    setup_logging(args.log_level)
    
    if args.test or args.test_all:
        logging.info("Running tests with log level: %s", args.log_level)
        test()
    else:
        logging.info("Running main program with log level: %s", args.log_level)
        main()
