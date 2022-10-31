from EduNLP.Pipeline import pipeline
from EduNLP.Pipeline.components import PREPROCESSING_PIPES
from EduNLP.ModelZoo.rnn import ElmoLMForPropertyPrediction, ElmoLMForKnowledgePrediction
from EduNLP.Pretrain import ElmoTokenizer
import pytest


class TestPipelines:
    def test_preprocessing_pipeline(self):
        item = "如图所示，则三角形ABC的面积是_。"
        items = [item, item]
        processor = pipeline(preprocess=['is_sif', 'to_sif', 'is_sif', 'seg_describe'])
        processor.add_pipe(name='seg', symbol='fm', before='seg_describe')
        assert processor.component_names == ['is_sif', 'to_sif', 'is_sif', 'seg', 'seg_describe']
        assert len(processor.component_names) == len(processor.pipeline)
        processor(item)
        processor(items)

        processor.remove_pipe(0)
        processor.remove_pipe('to_sif')
        processor.rename_pipe(1, 'seg_pipe')
        assert processor.component_names == ['is_sif', 'seg_pipe', 'seg_describe']

        processor = pipeline(preprocess=[item for item in PREPROCESSING_PIPES.keys()])
        processor(item)
        assert len(processor.component_names) == len(PREPROCESSING_PIPES)

    def test_error_usage_pipeline(self, pretrained_elmo_for_property_prediction_dir):
        pretrained_elmo_for_pp = ElmoLMForPropertyPrediction.from_pretrained(
            pretrained_elmo_for_property_prediction_dir)
        pretrained_elmo_tokenizer = ElmoTokenizer.from_pretrained(pretrained_elmo_for_property_prediction_dir)
        processor = pipeline(preprocess=['is_sif', 'seg_describe'])
        with pytest.raises(ValueError):
            pipeline(preprocess=['hallo', 'Welt'])
        with pytest.raises(ValueError):
            processor.add_pipe('to_sif', before='is_sif', after='is_sif')
        with pytest.raises(ValueError):
            processor.add_pipe('to_sif', before=-1)
        with pytest.raises(ValueError):
            processor.add_pipe('to_sif', after=4)
        with pytest.raises(ValueError):
            processor.add_pipe('to_sif', before='missing_pipe')
        with pytest.raises(ValueError):
            processor.add_pipe('to_sif', after='missing_pipe')
        with pytest.raises(ValueError):
            processor.add_pipe('to_sif', first=True, last=True)
        with pytest.raises(ValueError):
            processor.rename_pipe('Im_not_here', 'got_you')
        with pytest.raises(ValueError):
            processor.add_pipe('pipe_running_away')
        with pytest.raises(RuntimeError):
            pipeline()
        with pytest.raises(RuntimeError):
            pipeline(task='property-prediction', tokenizer=pretrained_elmo_tokenizer)
        with pytest.raises(RuntimeError):
            pipeline(model=pretrained_elmo_for_pp)
        with pytest.raises(KeyError):
            pipeline(task='I-am-a-robot')
        with pytest.raises(KeyError):
            pipeline(task='property-prediction', model=1, tokenizer=2)
        with pytest.raises(ValueError):
            p = pipeline(task='knowledge-prediction', model=pretrained_elmo_for_pp, tokenizer=pretrained_elmo_tokenizer)
            p("如图所示，则三角形ABC的面积是_。")

    # def test_property_prediction_pipeline_with_default_model(self):
    #     item = "如图所示，则三角形ABC的面积是_。"
    #     items = [item, item]
    #     processor = pipeline(task="property-prediction")
    #     assert 0 <= processor(item) <= 1
    #     assert len(processor(items)) == 2
    #     processor = pipeline(task="property-prediction", preprocess=['is_sif', 'seg_describe'])
    #     assert 0 <= processor(item) <= 1
    #     assert len(processor(items)) == 2

    def test_property_prediction_pipeline_with_specified_model(self, pretrained_elmo_for_property_prediction_dir):
        item = "如图所示，则三角形ABC的面积是_。"
        items = [item, item]
        pretrained_elmo_for_pp = ElmoLMForPropertyPrediction.from_pretrained(
            pretrained_elmo_for_property_prediction_dir)
        pretrained_elmo_tokenizer = ElmoTokenizer.from_pretrained(pretrained_elmo_for_property_prediction_dir)
        pretrained_elmo_for_pp.eval()
        processor = pipeline(task="property-prediction",
                             model=pretrained_elmo_for_pp,
                             tokenizer=pretrained_elmo_tokenizer)
        assert 0 <= processor(item)["property"] <= 1
        assert len(processor(items)) == 2

        processor.add_pipe("is_sif")
        assert processor.component_names == ["is_sif", "tokenizer", "ElmoLMForPropertyPrediction",
                                             "property-prediction"]
        assert len(processor) == 4
        processor.remove_pipe(0)
        assert processor.component_names == ["tokenizer", "ElmoLMForPropertyPrediction", "property-prediction"]
        assert len(processor.component_names) == len(processor.pipeline)

    # def test_knowledge_prediction_pipeline_with_default_model(self):
    #     item = "如图所示，则三角形ABC的面积是_。"
    #     items = [item, item]
    #     processor = pipeline(task="knowledge-prediction")
    #     assert 0 <= processor(item) <= 1
    #     assert len(processor(items)) == 2
    #     processor = pipeline(task="knowledge-prediction", preprocess=['is_sif', 'seg_describe'])
    #     assert 0 <= processor(item) <= 1
    #     assert len(processor(items)) == 2

    def test_knowledge_prediction_pipeline_with_specified_model(self, pretrained_elmo_for_knowledge_prediction_dir):
        item = "如图所示，则三角形ABC的面积是_。"
        items = [item, item]
        pretrained_elmo_for_kp = ElmoLMForKnowledgePrediction.from_pretrained(
            pretrained_elmo_for_knowledge_prediction_dir)
        pretrained_elmo_tokenizer = ElmoTokenizer.from_pretrained(pretrained_elmo_for_knowledge_prediction_dir)
        pretrained_elmo_for_kp.eval()
        processor = pipeline(task="knowledge-prediction",
                             model=pretrained_elmo_for_kp,
                             tokenizer=pretrained_elmo_tokenizer)
        assert isinstance(processor(item)["knowledge_list"], list)
        assert isinstance(processor(item)["knowledge_scores"], list)
        assert len(processor(items)[0]["knowledge_list"]) == len(processor(items)[1]["knowledge_list"])
        processor.add_pipe("is_sif")
        assert processor.component_names == ["is_sif", "tokenizer", "ElmoLMForKnowledgePrediction",
                                             "knowledge-prediction"]
