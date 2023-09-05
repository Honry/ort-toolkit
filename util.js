function compare(actual, expected, epsilon = 1e-3) {
  try {
    areCloseObjects(actual, expected, epsilon);
  } catch (e) {
    return false;
  }
  return true;
}

function areCloseObjects(actual, expected, epsilon) {
  let actualKeys = Object.getOwnPropertyNames(actual);
  let expectedKeys = Object.getOwnPropertyNames(expected);
  if (actualKeys.length != expectedKeys.length) {
    throw new Error(`Actual length ${actualKeys.length} not equal Expected length ${expectedKeys.length}`);
  }
  for (let i = 0; i < actualKeys.length; i++) {
    let key = actualKeys[i];
    let isArray = isTypedArray(actual[key]) && isTypedArray(expected[key]);
    let isObject = typeof (actual[key]) === 'object' && typeof (expected[key]) === 'object';
    if (isArray) {
      areCloseArrays(actual[key], expected[key], epsilon);
    } else if (isObject) {
      areCloseObjects(actual[key], expected[key], epsilon);
    } else {
      if (!areClosePrimitives(actual[key], expected[key])) {
        throw new Error(`Objects differ: actual[${key}] = ${JSON.stringify(actual[key])}, expected[${key}] = ${JSON.stringify(expected[key])}!`);
      }
    }
  }
  return true;
}

function areCloseArrays(actual, expected, epsilon) {
  let checkClassType = true;
  if (isTypedArray(actual) || isTypedArray(expected)) {
    checkClassType = false;
  }
  if (isTypedArray(actual) && isTypedArray(expected)) {
    checkClassType = true;
  }
  if (checkClassType) {
    const aType = actual.constructor.name;
    const bType = expected.constructor.name;

    if (aType !== bType) {
      throw new Error(`Arrays are of different type. Actual: ${aType}. Expected: ${bType}`);
    }
  }

  const actualFlat = isTypedArray(actual) ? actual : flatten(actual);
  const expectedFlat = isTypedArray(expected) ? expected : flatten(expected);

  if (actualFlat.length !== expectedFlat.length) {
    throw new Error(
      `Arrays have different lengths actual: ${actualFlat.length} vs ` +
      `expected: ${expectedFlat.length}.\n` +
      `Actual:   ${actualFlat}.\n` +
      `Expected: ${expectedFlat}.`);
  }
  for (let i = 0; i < expectedFlat.length; ++i) {
    const a = actualFlat[i];
    const e = expectedFlat[i];

    if (!areClosePrimitives(a, e)) {
      throw new Error(
        `Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
        `Actual:   ${actualFlat}.\n` +
        `Expected: ${expectedFlat}.`);
    }
  }
}

function areClosePrimitives(actual, expected, epsilon) {
  if (!isFinite(actual) && !isFinite(expected)) {
    return true;
  } else if (isNaN(actual) || isNaN(expected)) {
    return false;
  }

  const error = Math.abs(actual - expected);
  if (Math.abs(actual) >= 1) {
    if ((error > 1e-1) || error / Math.min(Math.abs(actual), Math.abs(expected)) > epsilon) {
      return false;
    }
  } else {
    if (error > epsilon) {
      return false;
    }
  }
  return true;
}

function isTypedArray(object) {
  return ArrayBuffer.isView(object) && !(object instanceof DataView);
}
