from EduNLP import logger
from typing import Union, List, Callable, Optional, Dict
from .components import TOKENIZER_PIPES
from ..Pretrain import PretrainedEduTokenizer
from ..ModelZoo.base_model import BaseModel

GenericTensor = Union[List["GenericTensor"], "torch.Tensor"]


class TokenizerPipeline(object):
    """
    A pipeline for tokenization processing.

    Parameters
    ----------
    pipe_names: `str` or `List[str]`, optional
        The quickly initialized pipeline components. For availabel pipes, check TOKENIZE_PIPES in `components`.
        To add componets more flexiblely with specific arguments or custom name, use `add_pipe`.

    Examples
    ----------
    >>> tkn = TokenizerPipeline(['is_sif', 'to_sif', 'is_sif', 'seg_describe'])
    >>> tkn.add_pipe(name='seg', symbol='fm', before='seg_describe')
    >>> tkn.component_names
    ['is_sif', 'to_sif', 'is_sif', 'seg', 'seg_describe']
    >>> item = "如图所示，则三角形ABC的面积是_。"
    >>> tkn(item)
    False
    True
    {'t': 3, 'f': 1, 'g': 0, 'm': 1}
    ['如图所示，则三角形', '[FORMULA]', '的面积是', '[MARK]', '。']
    """

    def __init__(self,
                 pipe_names: Optional[Union[List[str], str]] = None
                 ):
        self._tokenize_components = {}
        self.component_pipeline = []
        if isinstance(pipe_names, str):
            self.component_pipeline.append(pipe_names)
            self._tokenize_components[pipe_names] = TOKENIZER_PIPES[pipe_names]()
        elif isinstance(pipe_names, list) and len(pipe_names) > 0:
            if any(comp_name not in TOKENIZER_PIPES for comp_name in pipe_names):
                logger.error('Some components not existed!')
                raise ValueError
            for pipe_name in pipe_names:
                if pipe_name not in self._tokenize_components:
                    self._tokenize_components[pipe_name] = TOKENIZER_PIPES[pipe_name]()
            self.component_pipeline += [comp_name for comp_name in pipe_names]

    def __call__(self, inputs):
        for name in self.component_pipeline:
            proc = self._tokenize_components[name]
            try:
                inputs = proc(inputs)
            except Exception as e:
                logger.error(e)
        return inputs

    def __len__(self):
        return len(self.component_pipeline)

    def _get_pipe_index(
            self,
            before: Optional[Union[str, int]] = None,
            after: Optional[Union[str, int]] = None,
            first: Optional[bool] = None,
            last: Optional[bool] = None
    ):
        if sum(arg is not None for arg in [before, after, first, last]) > 1:
            logger.error('Only one of before/after/first/last can be set!')
            raise ValueError
        if last or not any(arg is not None for arg in [before, after, first]):
            return len(self)
        elif first:
            return 0
        elif isinstance(before, str):
            if before not in self.component_pipeline:
                logger.error('The before pipe does not exists!')
                raise ValueError
            else:
                return self.component_pipeline.index(before)
        elif isinstance(before, int):
            if before < 0 or before >= len(self.component_pipeline):
                logger.error('The before index must be greater than 0 and less than current length!')
            else:
                return before
        elif isinstance(after, str):
            if after not in self.component_pipeline:
                logger.error('The after pipe does not exists!')
                raise ValueError
            else:
                return self.component_pipeline.index(after) + 1
        elif isinstance(after, int):
            if after < 0 or after >= len(self.component_pipeline):
                logger.error('The after index must be greater than 0 and less than current length!')
            else:
                return after + 1
        else:
            raise ValueError

    def add_pipe(
            self,
            name: str,
            component: Optional[Callable] = None,
            before: Optional[Union[str, int]] = None,
            after: Optional[Union[str, int]] = None,
            first: Optional[bool] = None,
            last: Optional[bool] = None,
            *args,
            **kwargs
    ):
        """
        Add a component to the tokenization pipeline.
        Valid component must be Callable and feat its next component. Only one parameter of before/after/first/last
        can be set. Default setting is `last`.
        Notice:
        1. Please try to avoid more than one usages of one same pipe, otherwise you can only modify them with index.
            i.e. `before` and `after` works well only when the pipe is unique.
        2. The `*args, **kwargs` parameters will be passed to component constructor in `TOKENIZER_PIPES`, and this only
            works when you do not give a callable component.

        Parameters
        ----------
        name: `str`, required
            the name of pipe
        component: `Callable`, optional
            the custom pipe component, be careful with its nearest components' input&output.
        before: `str` or `int`, optional
            name or index of the component to insert new component directly before. Index start from 0.
        after: `str` or `int`, optional
            name or index of the component to insert new component directly after. Index start from 0.
        first: `bool`, optional
            if true, insert the component first in the pipeline.
        last: `bool`, optional
            if true, insert the component last in the pipeline.
        """
        pipe_index = self._get_pipe_index(before, after, first, last)
        if component is None and name not in self._tokenize_components:
            if name not in TOKENIZER_PIPES:
                logger.error(f'Unknown pipe "{name}"')
                raise ValueError
            else:
                self._tokenize_components[name] = TOKENIZER_PIPES[name](*args, **kwargs)
        else:
            if name in self._tokenize_components:
                logger.warn(f'One preserved component "{name}" has been replaced')
            self._tokenize_components[name] = component
        self.component_pipeline.insert(pipe_index, name)

    def remove_pipe(
            self,
            pipe: Union[str, int]
    ):
        if isinstance(pipe, str):
            if pipe not in self._tokenize_components:
                logger.error(f'Unknown pipe "{pipe}"')
                raise ValueError
            self.component_pipeline.remove(pipe)
            return self._tokenize_components[pipe]
        else:
            removed = self.component_pipeline.pop(pipe)
            return self._tokenize_components[removed]

    def rename_pipe(
            self,
            old_pipe: Union[str, int],
            new_name: str,
    ):
        """
        Rename a component from the tokenizer pipeline.

        Parameters
        ----------
        old_pipe: `str` or `int`, required
            old component name for `str`, or old component index in the pipeline for `int`
        new_name: `str`, required
            new name for the component
        """
        if isinstance(old_pipe, int):
            old_pipe = self.component_pipeline[old_pipe]
        if old_pipe not in self._tokenize_components:
            logger.error(f'Unknown pipe "{old_pipe}"')
            raise ValueError
        self._tokenize_components[new_name] = self._tokenize_components.pop(old_pipe)
        self.component_pipeline = list(map(lambda x: x.replace(old_pipe, new_name), self.component_pipeline))

    @property
    def component_names(self):
        """
        Get the names of pipeline components
        """
        return self.component_pipeline.copy()

    @property
    def pipeline(self):
        """
        Get the processing pipeline consisting of tuple (name, component) tuples.
        """
        return [(name, self._tokenize_components[name]) for name in self.component_pipeline]
