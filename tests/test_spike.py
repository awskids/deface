import onnx
import onnx.parser

# import onnxruntime
import pytest

from onnx import helper, shape_inference


onnx_input_name = "input.1"
onnx_output_names = ["537", "538", "539", "540"]


@pytest.fixture
def ONNX_PATH():
    return "./deface/centerface.onnx"


@pytest.fixture
def MODEL(ONNX_PATH):
    static_model = onnx.load(ONNX_PATH)
    return static_model


def test_review_model(MODEL):
    static_model = MODEL
    input_dims, output_dims = {}, {}
    for node in static_model.graph.input:
        dims = [d.dim_value for d in node.type.tensor_type.shape.dim]
        input_dims[node.name] = dims
    for node in static_model.graph.output:
        dims = [d.dim_value for d in node.type.tensor_type.shape.dim]
        output_dims[node.name] = dims

    # onnx_model = onnx.parser.parse_model(static_model)
    inferred_model = shape_inference.infer_shapes(static_model)

    # print(inferred_model)
    # print("input_dims", input_dims.keys())
    print("input_dims", input_dims["input.1"])
    print("output_dims", output_dims)

    # in_shape (1280, 720) ==> < 720, 1280 >
    # model definition.
    # input_dims [10, 3, 32, 32]
    # output_dims {'537': [10, 1, 8, 8], '538': [10, 2, 8, 8], '539': [10, 2, 8, 8], '540': [10, 10, 8, 8]}
    # You can see the downsampling in dimensions

    # after modified...
    # input ['B', 3, 'H', 'W']
    # ------------------------  1280 >> 320 = 1/4
    # after dynamic update to model
    # heatmap (1, 1, 184, 320)
    # scale (1, 2, 184, 320)
    # offset (1, 2, 184, 320)
    # lms (1, 10, 184, 320)


def test__review_diagram(MODEL):
    from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

    model_onnx = MODEL

    pydot_graph = GetPydotGraph(
        model_onnx.graph,  # model_onnx is a ModelProto instance
        name=model_onnx.graph.name,
        rankdir="TP",
        node_producer=GetOpNodeProducer("docstring"),
    )
    pydot_graph.write_dot("graph.dot")


# def test__convert_onnx_to_tf():
#     from onnx_tf.backend import prepare
#     pass
