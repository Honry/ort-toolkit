const models = {
  // tjs/albert-base-v2/onnx/model.onnx
  // TODO: NaN
  'albert-base-v2': 'bert64',

  // tjs/facebook/bart-large-cnn/onnx/encoder_model.onnx
  'bart-large-cnn-encoder': 'bert64',

  // tjs/facebook/detr-resnet-50/onnx/model.onnx
  // TODO: conformance fails
  'detr-resnet-50': 'img224',

  // tjs/t5-small/onnx/encoder_model.onnx
  't5-small-encoder': { 'input_ids': ['int64', 99n, [1, 128]] },

  // tjs/bert-base-uncased/onnx/model.onnx
  'bert-base-uncased': 'bert64',

  // openai/whisper-tiny/onnx/decoder_model_merged.onnx
  'whisper-tiny-decoder': 'whisper-tiny-decoder',

  // openai/whisper-tiny/onnx/encoder_model.onnx
  'whisper-tiny-encoder': { 'input_features': ['float32', 'random', [1, 80, 3000]] },




  // TODO
  'densenet-9': 'img224',
  'efficientnet-lite4-11': { 'images:0': ['float32', 'random', [1, 224, 224, 3]] },
  'emotion-ferplus-8': { 'Input3': ['float32', 'random', [1, 1, 64, 64]] },
  'mobilenetv2-7': 'img224',
  'mobilenetv2-10': 'img224',
  'mobilenetv2-12': 'img224',


  'tinyyolov2-8': { 'image': ['float32', 'random', [1, 3, 416, 416]] },

  // could not find
  'candy-8': 'img224', // If the value is set to 0.5, conformance test would fail.
  'resnet50-v1-12': 'img224',
  'resnet50-v2-7': 'img224',

  // todo


  'inception-v1-12': {
    'data_0': ['float32', 0.5, [1, 3, 224, 224]],
  },




  'segment-anything-vit-h-static-shapes-origin-im-size-initializer-optimized-float32': {
    'image_embeddings': ['float32', 0.5, [1, 3, 224, 224]],
    'point_coords': ['float32', [327.1111, 426.875, 241.77777, 341.5], [1, 2, 2]],
    'point_labels': ['float32', [0, 1], [1, 2]],
    'mask_input': ['float32', 0, [1, 1, 256, 256]],
    'has_mask_input': ['float32', 1, [1]],
  },

  'squeezenet1.1-7': {
    'data': ['float32', 0.5, [1, 3, 224, 224]],
  },

  // 't5-encoder-12'



  'yolo': {
    'image': ['float32', 0.5, [1, 3, 416, 416]],
  },

  // no corresponding models
  'sam-h': {
    'image_embeddings': ['float32', 0.5, [1, 256, 64, 64]],
    'point_coords': ['float32', [327.1111, 426.875, 241.77777, 341.5], [1, 2, 2]],
    'point_labels': ['float32', [0, 1], [1, 2]],
    'mask_input': ['float32', 0, [1, 1, 256, 256]],
    'has_mask_input': ['float32', 1, [1]],
    // orig_im_size is optimized out for this model
    //'orig_im_size': ['float32', [2], [1200., 1800.]],
  },

  // TODO: Convert to actual float16 values. We're just estimating perf with this one, not correctness.
  'sam-h-16': {
    'image_embeddings': ['float16', 0.5, [1, 256, 64, 64]],
    'point_coords': ['float16', [327, 426, 241, 341], [1, 2, 2]],
    'point_labels': ['float16', [0, 2], [1, 2]],
    'mask_input': ['float16', 0, [1, 1, 256, 256]],
    'has_mask_input': ['float16', 1, [1]],
    // orig_im_size is optimized out for this model
    //'orig_im_size': ['float16', [1200., 1800.], [2]],
  },

  'onnx-add': {
    'A': ['float32', 1, [5]],
    'B': ['float32', 1, [5]],
  }
}

function getFeeds(session, model) {
  let feeds = {};
  let inputs = models[model];
  let inputNames = session.inputNames;

  if (['bart-large', 'bart-large-12'].indexOf(inputs) >= 0) {
    let enc_seqlen = 128;
    const kvdim = (model === 'bart-large') ? 16 : 12;
    const hiddendim = (model === 'bart-large') ? 1024 : 768;
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values')) {
        feeds[v] = genTensor('float32', 1., [1, kvdim, seqlen, 64]);
      }
      if (v.startsWith('encoder_attention_mask')) {
        feeds['encoder_attention_mask'] = genTensor('int64', 1n, [1, enc_seqlen]);
      }
    }
    feeds['use_cache_branch'] = genTensor('bool', false, [1]);
    feeds['input_ids'] = genTensor('int64', 99n, [1, seqlen]);
    feeds['encoder_hidden_states'] = genTensor('float32', 1, [1, enc_seqlen, hiddendim]);
    return feeds;
  }


  if (['bert', 'bert64'].indexOf(inputs) >= 0) {
    let seqlen;
    if ([].indexOf(model) >= 0) {
      seqlen = 1;
    } else {
      seqlen = 128;
    }
    const dtype = inputs == 'bert' ? 'int32' : 'int64';
    const value = inputs == 'bert' ? 99 : 99n;
    const one = inputs == 'bert' ? 1 : 1n;

    for (var k in inputNames) {
      const v = inputNames[k];
      if (v === 'input_ids') {
        feeds[v] = getTensor(dtype, value, [1, seqlen]);
      }
      if (v === 'input_mask' || v === 'attention_mask') {
        feeds[v] = getTensor(dtype, one, [1, seqlen]);
      }
      if (v === 'token_type_ids' || v == 'segment_ids') {
        feeds[v] = getTensor(dtype, one, [1, seqlen]);
      }
    }
    return feeds;
  }

  if (model === 'whisper-decoder') {
    feeds['input_ids'] = genTensor('int64', 1n, [1, 1]);
    feeds['encoder_hidden_states'] = genTensor('float32', 'random', [1, 1500, 384]);
    for (var k in inputNames) {
      const v = inputNames[k];
      if (v.startsWith('past_key_values.')) {
        if (v.includes('decoder')) {
          feeds[v] = genTensor('float32', 1, [1, 6, seqlen, 64]);
        } else if (v.includes('encoder')) {
          feeds[v] = genTensor('float32', 1, [1, 6, 1500, 64]);
        }
      }
    }
    feeds['use_cache_branch'] = genTensor('bool', false, [1]);
    return feeds;
  }


  if (inputs === 'img224') {
    feeds[inputNames[0]] = getTensor('float32', 'random', [1, 3, 224, 224]);
    return feeds;
  }

  for (let key in inputs) {
    let value = inputs[key];
    feeds[key] = getTensor(value[0], value[1], value[2]);
  }
  return feeds;
}


function getTensor(type, data, dims) {
  let typedArray;
  if (type === 'bool') {
    return new ort.Tensor(type, data, [1]);
  } else if (type === 'uint16') {
    typedArray = Uint16Array;
  } else if (type === 'float16') {
    typedArray = Uint16Array;
  } else if (type === 'float32') {
    typedArray = Float32Array;
  } else if (type === 'int32') {
    typedArray = Int32Array;
  } else if (type === 'int64') {
    typedArray = BigInt64Array;
  }

  let _data;
  if (Array.isArray(data)) {
    _data = data === 'random' ? Math.random() : data;
  } else {
    let size = 1;
    dims.forEach((dim) => {
      size *= dim;
    });
    _data = typedArray.from({ length: size }, () => data === 'random' ? Math.random() : data);
  }
  return new ort.Tensor(type, _data, dims);
}