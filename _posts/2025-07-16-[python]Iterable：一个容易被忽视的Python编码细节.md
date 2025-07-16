## Type hints
近年来，越来越多的 Python 开发者愿意为变量声明类型了，变化非常明显。

```python
def add(left, right):
    return left + right
```

```python
from typing import TypeVar, Union

T = TypeVar('T', int, float)

def add_typed(left: T, right: T) -> T:
    return left + right
```

虽然 type hints 并不会在运行时进行类型检查，但它们足以让 IDE 在运行前就报出不少类型风险，也让阅读代码的人有了更多思考空间，不至于迷失在“这到底是什么类型”的疑惑中。

## Iterator 和 Iterable 的问题

Python 对 Iterator 的要求只有一条：

> An iterator object implements `__next__`, which is expected to return the next element of the iterable object that returned it, and to raise a `StopIteration` exception when no more elements are available.
> 
> 一个迭代器对象需要实现 `__next__` 方法，该方法应当返回其所属可迭代对象的下一个元素；当没有更多元素可返回时，应抛出 `StopIteration` 异常。
>
> -- [Python Wiki](https://wiki.python.org/moin/Iterator)

Python 对 Iterable 的要求也很简单：

> An iterable object is an object that implements `__iter__`, which is expected to return an iterator object. 
>
> 一个可迭代对象是实现了 `__iter__` 方法的对象，该方法应当返回一个迭代器对象。
>
> -- [Python Wiki](https://wiki.python.org/moin/Iterator)

这两年我还写了不少 Rust。在 Rust 里，Python 的 `Iterator` 对应 [`std::iter::Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html)，`Iterable` 则对应 [`std::iter::IntoIterator`](https://doc.rust-lang.org/std/iter/trait.IntoIterator.html)。正是有了写 Rust 的经验，我才能一眼看出下面代码的问题：

```python
def stream(
        self,
        *,
        # ...
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        # ...
    ) -> AsyncChatCompletionStreamManager[ResponseFormatT]:
    # ...
    _validate_input_tools(tools)
    # ...
    api_request = self.create(
            ...
            tools=tools,
            ...
        )
        return AsyncChatCompletionStreamManager(
            ...
            input_tools=tools,
        )
```

这段代码节选自 OpenAI 的 Python 接口 [completions.py](https://github.com/openai/openai-python/blob/859b4db4a7b3c229cd4c19eb21642faca007530b/src/openai/resources/chat/completions/completions.py#L1479)。问题出现在 [#1129](https://github.com/openai/openai-python/pull/1129) 这个 PR。

作为 Rust 选手的直觉告诉我，一个 Iterable 在这里被用到了三次，这就要求这个 Iterable 在多次迭代时本身状态不可变，否则无法保证三次迭代的结果一致。

通俗点说：这个 Iterable 必须保证多次调用 `__iter__` 返回的迭代器，其迭代结果是一致的。

验证一下。对于普通容器当然没问题，但对于[生成器](https://wiki.python.org/moin/Generators)就大有问题。生成器作为迭代器，其状态是会变的，所以三次调用的结果无法保证一致。

```python
from typing import Iterable, List

def generate_from_list(lst: List[int]):
    for item in lst:
        yield item

# 容器可以多次迭代，产生相同结果
iterable0 = [1 ,2 ,3]
assert list(iterable0) == list(iterable0) == [1, 2, 3]

# 生成器迭代器只能迭代一次
iterable1 = generate_from_list([1, 2, 3])
assert list(iterable1) == [1, 2, 3]
assert list(iterable1) == []

# 容器外套一层map并不能保持容器可以多次迭代的特征，只能迭代一次
iterable2 = map(lambda x: x + 1, [1, 2, 3])
assert list(iterable2) == [2, 3, 4]
assert list(iterable2) == []

# 生成器额迭代器外套一层map自不必多说，只能迭代一次
iterable3 = map(lambda x: x + 1, generate_from_list([1, 2, 3]))
assert list(iterable3) == [2, 3, 4]
assert list(iterable3) == []
```

运行结果：

<iframe width="100%" height="500px" src="https://godbolt.org/e?readOnly=false&hideEditorToolbars=true#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AAHAJ4EEA0kvrICeAZUboAwqloBXFgwkBWUs4AMngMmAByXgBGmMQSXKTGqAqEDgxunt5%2BCUkpAsGhESzRsVzxNph2qUIETMQE6V4%2BXP7llQLVtQT54VExcdY1dQ2ZzQOd3YXFcQCU1qgexMjsHFTEqCwA1ASmxiHAG3gsiXUbAJIEMUyR9KQbwQoEAKQaAILPL1hUG8CMlxcA%2Bqt1v9aHgHhBaA8QHcwU9fAAhEJwgAi0xA7w2mI2/GIBwumxCG0hBHRryx5I2pjwmFo6DxmBY7yZACYAMwbR7YXyPF7YR7wgCcnO5AqFLwAHMLHuKAGI8uXYST8%2BE87lcnlKgVvbAANh5Lh5XE5kvhcpeyM5SvhKperM5cvhBvFBsV/LeLwA7JyvQLNQqfSr4ZL1QKNNKXV7rY8BXbddG%2BQKDa9CJdrpgNBzWciOQiuBtSMz83bfBbXkwFEoTqDwSniFd6BpppmLVmibCILX6%2Bmm48s73s49c7dmbdi6W3q9HmyOdgfX7OXrxZLQ1LRTzgyazRbXVHbauxcHubKedrJeK7fDt5uedurTbY1az2G4y8DS8jcmLnW03n%2B99fnWAJAiwILtoO8LxBsI4bGO0zvOWlYEG2NZfl2XA9n2rbgZB0FjvBFYxEh1YEB2qE/hhLYDgig7jiy7JSry/JCiGB4MZqerqi8IpejuT5ShB0rMiwTDGPxQrwhxj7Sle0oXre/JygK26Lk6RqcXygr7uuUrHi8Cp3qqbHRtqeqvoaxoKTelrKjysY%2Bm84pKrO0ricx3Lwny4pqQ6ToukeZqnrJ/IyaaVk7ve1lBtKz6mW%2BH4vJ2aaFn%2BwnGBCrCROgTAbOoOUcsy8IbJB2HDqONFwWWBFVu2iX0MyFF/uBuG3EqJb4YhyEkbVmD1c2jXUW1k6TtO3rRvOcZLtGz4iqx2Bisy0YhVukURSx2mcexDHcZFfHqgJ4pCSJFnil6LzalJ4rbu5crimq3IvJqgV6fyTk%2BeGOkBcdckWaF5orbZO3RQu%2BrmZ%2Bqb0PRrapelRRZTl0KqPlhWQT8oSAZggJrCBxEQCVUFlSW0wVS8CGEZ1pHg5grINVhCLNRsrXjqT1UoZT1N9bTKolhwsy0Jwvi8D4HBaKQqCcMo5iWAwGwKPMiyYPlrI8KQJLCzzswANYgDqGgAHS%2BMyzJcDGRtSEuzL%2BHzHCSILmi8GLHC8AoIAaCrduzHAsBIJgqgVB4RBkBQEC1MACjKIYNJCJYADuQvK2gRx0Ew9gCOHoS0FHqCx3bpAJ7s9CxGYFgCErrK5%2Bs%2BcxGErDLLwed0DEADy/uZ9nasBL7yAvMQoecLwPsVNU%2BBC7w/CCCIYjsFIMiCIoKjqO3ujxAYRggEXUv6HgkTO5AsyoMYKcMM7jtzAsSx6A8w9p5HMdx7w0d1sYnA8Lz/O2%2B3DvYJ3/skBsEvF9LVkusuDsggP/DeGxcCEF/lOJW0xeCqy0ETUgCBMBMCwLECAmsQCsgFLrSQHoPTig0DqIht1SgektpwG2pAhYiwdk7F2bs1bIKtsyd%2B9C%2B7MKQbMAAbjEZIjhJBAA%3D%3D"></iframe>

## 反思

我不反思，我一眼就看出这个问题了，我为啥要反思。但需要反思的人还真不少。

写代码时一定要清楚，自己进行的每一个操作对前置操作有什么依赖，是否修改了某个状态，这个状态是否应该由这个操作修改。如果不确定，就必须按照最保守的策略来。

还是以 `Iterable` 为例，拿到一个可迭代对象后，如果不确定它是不是容器、能否多次迭代且结果一致，那就只允许自己对它做一次迭代。如果因为种种原因不得不多次迭代，那就把它转成 `list`，再在这个 `list` 上反复迭代。

[#1129](https://github.com/openai/openai-python/pull/1129) 的错误略有不同，它不是在写新代码时出的问题，而是在放宽已有代码的限制时，只考虑了容器的情况，没考虑到其他典型可迭代对象（生成器、map 等）。这个 PR 合并一年多了，似乎还没人踩坑，这恰恰说明大多数场景下传入的就是容器（List），原 PR 放宽约束意义不大，反而引入了潜在风险。

[#1606](https://github.com/openai/openai-python/pull/1606/files#diff-119a3011ffba204040c083a78bf5ad392816d0e3ca2610977ce50de95dcafbf8R39-R45) 的错误则更直接。一个 `validate_input_tools` 函数本不该对输入数据做任何更改，但根据刚才的讨论，这里有潜在的对 `Iterable` 状态的修改，所以这个函数的类型标注就应该只接受容器，或者其他能表示**不可变**可迭代对象的类型。

## 总结

总之，`Iterable` 虽然是 Python 类型标注中常见的一个概念，但它的“可多次迭代且结果一致”这一点很容易被忽视。写代码时，尤其是在类型标注、接口设计和数据处理时，一定要明确区分“容器型可迭代对象”和“一次性可迭代对象”，避免踩坑。遇到不确定的情况，最保险的做法就是把它转成 `list`，这样既保证了多次迭代的一致性，也让代码的行为更加可控和易于理解。

（🤔感觉这个还是不要做面试题了）