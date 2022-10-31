import torch
from EduNLP import logger
from typing import Union, List, Callable, Optional, Dict, Any, Tuple
from .components import PREPROCESSING_PIPES
from ..Pretrain import PretrainedEduTokenizer
from ..ModelZoo.base_model import BaseModel
from transformers.modeling_outputs import ModelOutput
from abc import ABC, abstractmethod

GenericTensor = Union["torch.Tensor", List["GenericTensor"]]


class PreProcessingPipeline(object):
    """
    A pipeline for tokenization processing.

    Parameters
    ----------
    pipe_names: `str` or `List[str]`, optional
        The quickly initialized pipeline components. For availabel pipes, check TOKENIZE_PIPES in `components`.
        To add componets more flexiblely with specific arguments or custom name, use `add_pipe`.

    Examples
    ----------
    >>> tkn = PreProcessingPipeline(['is_sif', 'to_sif', 'is_sif', 'seg_describe'])
    >>> tkn.add_pipe(name='seg', symbol='fm', before='seg_describe')
    >>> tkn.component_names
    ['is_sif', 'to_sif', 'is_sif', 'seg', 'seg_describe']
    >>> item = "如图所示，则三角形ABC的面积是_。"
    >>> tkn(item)
    False
    True
    {'t': 3, 'f': 1, 'g': 0, 'm': 1}
    ['如图所示，则三角形', '[FORMULA]', '的面积是', '[MARK]', '。']
    >>> tkn.rename_pipe(0, 'is_sif_lol')
    >>> tkn.add_pipe('to_sif', component=lambda x:x, first=True) # This won't succeed for the same name pipe exists
    >>> tkn.component_names
    ['is_sif_lol', 'to_sif', 'is_sif_lol', 'seg', 'seg_describe']
    """

    def __init__(self,
                 pipe_names: Optional[Union[List[str], str]] = None
                 ):
        self._preproc_components = {}
        self.component_pipeline = []
        if isinstance(pipe_names, list) and len(pipe_names) > 0:
            if any(comp_name not in PREPROCESSING_PIPES for comp_name in pipe_names):
                logger.error('Some components not existed!')
                raise ValueError
            for pipe_name in pipe_names:
                if pipe_name not in self._preproc_components:
                    self._preproc_components[pipe_name] = PREPROCESSING_PIPES[pipe_name]()
            self.component_pipeline += [comp_name for comp_name in pipe_names]

    def __call__(self, inputs):
        for name in self.component_pipeline:
            proc = self._preproc_components[name]
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
                raise ValueError
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
                raise ValueError
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
        2. The `*args, **kwargs` parameters will be passed to component constructor in `PREPROCESSING_PIPES`,
            and this only works when you do not give a callable component.

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
        if component is None and name not in self._preproc_components:
            if name not in PREPROCESSING_PIPES:
                logger.error(f'Unknown pipe "{name}"')
                raise ValueError
            else:
                self._preproc_components[name] = PREPROCESSING_PIPES[name](*args, **kwargs)
        else:
            if name in self._preproc_components:
                logger.warn(f'One preserved component "{name}" has the same name, inserting is stopped.')
                return
            self._preproc_components[name] = component
        self.component_pipeline.insert(pipe_index, name)

    def remove_pipe(
            self,
            pipe: Union[str, int]
    ):
        """
        Remove a component from the pre-processing pipeline
        """
        if isinstance(pipe, str):
            if pipe not in self._preproc_components:
                logger.error(f'Unknown pipe "{pipe}"')
                raise ValueError
            self.component_pipeline.remove(pipe)
            return self._preproc_components[pipe]
        else:
            removed = self.component_pipeline.pop(pipe)
            return self._preproc_components[removed]

    def rename_pipe(
            self,
            old_pipe: Union[str, int],
            new_name: str,
    ):
        """
        Rename a component from the pre-processing pipeline.

        Parameters
        ----------
        old_pipe: `str` or `int`, required
            old component name for `str`, or old component index in the pipeline for `int`
        new_name: `str`, required
            new name for the component
        """
        if isinstance(old_pipe, int):
            old_pipe = self.component_pipeline[old_pipe]
        if old_pipe not in self._preproc_components:
            logger.error(f'Unknown pipe "{old_pipe}"')
            raise ValueError
        self._preproc_components[new_name] = self._preproc_components.pop(old_pipe)
        self.component_pipeline = [new_name if i == old_pipe else i for i in self.component_pipeline]

    @property
    def component_names(self):
        """
        Get the names of pipeline components
        """
        return self.component_pipeline.copy()

    @property
    def pipeline(self):
        """
        Get the processing pipeline consisting of (name, component) tuples.
        """
        return [(name, self._preproc_components[name]) for name in self.component_pipeline]


class Pipeline(ABC):
    """
    The pipeline class is the class from which all pipelines inherit. Pipeline workflow is defined as a sequence of the
    following operations:

        Input -> PreProcessingPipeline -> Tokenization -> Model Inference
                                                    -> Post-Processing (downstream task dependent) -> Output

    This class is not for using directly, refer to `Pipeline.__init__.pipeline` function.
    """

    def __init__(self,
                 task: Optional[str] = None,
                 model: Optional[BaseModel] = None,
                 tokenizer: Optional[PretrainedEduTokenizer] = None,
                 preproc_pipe_names: Optional[List] = None,
                 **kwargs
                 ):
        if preproc_pipe_names is None:
            preproc_pipe_names = []
        self.preproc_pipeline = PreProcessingPipeline(preproc_pipe_names)
        self.tokenizer = tokenizer
        self.model = model
        self.task = task

        self._tokenize_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)

    def __len__(self):
        _length = len(self.preproc_pipeline) + sum(
            component is not None for component in [self.tokenizer, self.model, self.task])
        return _length

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters: Dict):
        """
        _sanitize_parameters will be automatically called with any excessive named arguments
        from either '__init__' and '__call__' methods.
        Any inheritor of Pipeline should implement it, and it should return 3 dictionaries of parameters used by
        '_tokenize', '_forward' and 'postprocess' methods
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def _tokenize(self, input_: Any, **tokenize_parameters: Dict) -> Dict[str, GenericTensor]:
        """
        _tokenize will take the `input_` of a specific pipeline, go through it on a tokenizer and return a dictionary
        of everything necessary for `forward` to run properly.
        """
        raise NotImplementedError("_tokenize not implemented")

    @abstractmethod
    def _forward(self, input_: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        """
        _forward will receive the prepared dictionary from `tokenization` and run it on the model.
        """
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        """
        postprocess will receive the outputs of `_forward` method and reformat them into something more friendly
        based on specific task.
        """
        raise NotImplementedError("postprocess not implemented")

    def add_pipe(self, *args, **kwargs):
        """
        refer to PreProcessingPipeline.add_pipe
        """
        return self.preproc_pipeline.add_pipe(*args, **kwargs)

    def remove_pipe(self, *args, **kwargs):
        """
        refer to PreProcessingPipeline.remove_pipe
        """
        return self.preproc_pipeline.remove_pipe(*args, **kwargs)

    def rename_pipe(self, *args, **kwargs):
        """
        refer to PreProcessingPipeline.rename_pipe
        """
        return self.preproc_pipeline.rename_pipe(*args, **kwargs)

    @property
    def component_names(self):
        """
        Get the names of pipeline components
        """
        _component_names = self.preproc_pipeline.component_names.copy() if len(self.preproc_pipeline) > 0 else []
        if self.tokenizer is not None:
            _component_names.append("tokenizer")
        if self.model is not None:
            _component_names.append(self.model.__class__.__name__)
        if self.task is not None:
            _component_names.append(self.task)
        return _component_names

    @property
    def pipeline(self):
        """
        Get the processing pipeline consisting of (name, component) tuples.
        """
        _pipeline = self.preproc_pipeline.pipeline if len(self.preproc_pipeline) > 0 else []
        if self.tokenizer is not None:
            _pipeline.append(("tokenizer", self.tokenizer))
        if self.model is not None:
            _pipeline.append((self.model.__class__.__name__, self.model))
        if self.task is not None:
            _pipeline.append((self.task, None))
        return _pipeline

    def __call__(self, inputs, *args, num_workers=None, **kwargs):
        if args:
            logger.warning(f"Ignoring args: {args}")
        is_batch = isinstance(inputs, list)

        tokenize_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        tokenize_params = {**self._tokenize_params, **tokenize_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        if is_batch:
            return [self.run_single(item, tokenize_params, forward_params, postprocess_params) for item in inputs]
        else:
            return self.run_single(inputs, tokenize_params, forward_params, postprocess_params)

    def run_single(self, inputs, tokenize_params, model_params, postprocess_params):
        tokenize_inputs = self.preproc_pipeline(inputs)
        model_inputs = self._tokenize([tokenize_inputs], **tokenize_params)
        model_outputs = self._forward(model_inputs, **model_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs
