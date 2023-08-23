onnx_config = dict(type='onnx',
                   export_params=True,
                   keep_initializers_as_inputs=False,
                   opset_version=11,
                   save_file='end2end.onnx',
                   input_names=['input'],
                   output_names=['dets', 'labels'],
                   input_shape=(640, 640),
                   optimize=True,
                   dynamic_axes={
                       'input': {
                           0: 'batch',
                           2: 'height',
                           3: 'width'
                       },
                       'dets': {
                           0: 'batch',
                           1: 'num_dets'
                       },
                       'labels': {
                           0: 'batch',
                           1: 'num_dets'
                       }
                   })
codebase_config = dict(type='mmyolo',
                       task='ObjectDetection',
                       model_type='end2end',
                       post_processing=dict(score_threshold=0.05,
                                            confidence_threshold=0.005,
                                            iou_threshold=0.5,
                                            max_output_boxes_per_class=200,
                                            pre_top_k=5000,
                                            keep_top_k=100,
                                            background_label_id=-1),
                       module=['mmyolo.deploy', 'mmyolo', 'yoloms'])
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=True, max_workspace_size=1 << 32),
    model_inputs=[
        dict(input_shapes=dict(input=dict(min_shape=[1, 3, 64, 64],
                                          opt_shape=[1, 3, 640, 640],
                                          max_shape=[1, 3, 1344, 1344])))
    ])
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
