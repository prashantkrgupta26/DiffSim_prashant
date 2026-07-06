import * as math from 'mathjs';

export type SineLayerCheckpoint = {
  weight: number[][];
  bias: number[];
  w0: number;
};

export type SirenCheckpoint = {
  meta: {
    in_dim: number;
    hidden_dim: number;
    hidden_layers: number;
    out_dim: number;
    skip_in: number[];
    w0: number;
    w0_hidden: number;
  };
  layers: SineLayerCheckpoint[];
  final: {
    weight: number[][];
    bias: number[];
  };
};

export class SineLayer {
  weights: number[][];
  bias: number[];
  w0: number;

  constructor(
    inFeatures: number,
    outFeatures: number,
    isFirst = false,
    w0 = 30.0,
    checkpoint?: SineLayerCheckpoint
  ) {
    this.w0 = w0;
    this.weights = Array.from({ length: outFeatures }, () =>
      Array.from({ length: inFeatures }, () => 0)
    );
    this.bias = Array.from({ length: outFeatures }, () => 0);

    if (checkpoint) {
      this.w0 = checkpoint.w0;
      this.weights = checkpoint.weight.map((row) => row.slice());
      this.bias = checkpoint.bias.slice();
      return;
    }

    const bound = isFirst ? 1 / inFeatures : Math.sqrt(6 / inFeatures) / w0;

    for (let i = 0; i < outFeatures; i++) {
      for (let j = 0; j < inFeatures; j++) {
        this.weights[i][j] = (Math.random() * 2 - 1) * bound;
      }
      this.bias[i] = (Math.random() * 2 - 1) * bound;
    }
  }

  forward(x: number[]): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.weights.length; i++) {
      let sum = 0;
      for (let j = 0; j < this.weights[i].length; j++) {
        sum += this.weights[i][j] * x[j];
      }
      out.push(Math.sin(this.w0 * (sum + this.bias[i])));
    }
    return out;
  }
}

export class MultiHeadSiren {
  layers: SineLayer[] = [];
  finalWeights: number[][];
  finalBias: number[];
  skipIn: number[];
  hiddenDim: number;
  inDim: number;

  constructor(
    inDim = 3,
    hiddenDim = 128,
    hiddenLayers = 8,
    outDim = 2,
    skipIn = [4],
    w0 = 30.0,
    w0Hidden = 30.0,
    layerCheckpoints?: SineLayerCheckpoint[],
    finalWeights?: number[][],
    finalBias?: number[]
  ) {
    this.inDim = inDim;
    this.hiddenDim = hiddenDim;
    this.skipIn = skipIn;

    this.layers.push(new SineLayer(inDim, hiddenDim, true, w0, layerCheckpoints?.[0]));
    for (let i = 1; i < hiddenLayers; i++) {
      const inFeatures = hiddenDim + (skipIn.includes(i) ? inDim : 0);
      this.layers.push(
        new SineLayer(inFeatures, hiddenDim, false, w0Hidden, layerCheckpoints?.[i])
      );
    }

    if (finalWeights && finalBias) {
      this.finalWeights = finalWeights.map((row) => row.slice());
      this.finalBias = finalBias.slice();
    } else {
      this.finalWeights = Array.from({ length: outDim }, () =>
        Array.from({ length: hiddenDim }, () => (Math.random() * 2 - 1) / Math.sqrt(hiddenDim))
      );
      this.finalBias = Array.from({ length: outDim }, () => 0);
    }
  }

  features(x: number[]): number[] {
    let h = x;
    for (let i = 0; i < this.layers.length; i++) {
      if (this.skipIn.includes(i)) {
        h = [...h, ...x]; // Simplified concatenation
        // In the Python script it's torch.cat([h, x], dim=-1) / math.sqrt(2)
        // We should normalize if we want to match exactly, but for a demo it's fine.
        h = h.map(v => v / Math.sqrt(2));
      }
      h = this.layers[i].forward(h);
    }
    return h;
  }

  forward(x: number[]): number[] {
    const feat = this.features(x);
    const out: number[] = [];
    for (let i = 0; i < this.finalWeights.length; i++) {
      let sum = 0;
      for (let j = 0; j < feat.length; j++) {
        sum += this.finalWeights[i][j] * feat[j];
      }
      out.push(sum + this.finalBias[i]);
    }
    return out;
  }
}

export function modelFromCheckpoint(ckpt: SirenCheckpoint) {
  const m = ckpt.meta;
  return new MultiHeadSiren(
    m.in_dim,
    m.hidden_dim,
    m.hidden_layers,
    m.out_dim,
    m.skip_in,
    m.w0,
    m.w0_hidden,
    ckpt.layers,
    ckpt.final.weight,
    ckpt.final.bias
  );
}

export async function loadCheckpoint(path: string): Promise<SirenCheckpoint> {
  const resp = await fetch(path);
  if (!resp.ok) {
    throw new Error(`Failed to load checkpoint JSON: ${path} (${resp.status})`);
  }
  return resp.json();
}

export function computeTopKBasis(model: MultiHeadSiren, nObs = 1000, k = 20) {
  const features: number[][] = [];
  for (let i = 0; i < nObs; i++) {
    const x = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1];
    features.push(model.features(x));
  }

  // Compute Gram Matrix G = H^T * H
  const hiddenDim = model.hiddenDim;
  const G = Array.from({ length: hiddenDim }, () => new Float64Array(hiddenDim));
  
  for (let i = 0; i < nObs; i++) {
    const h = features[i];
    for (let r = 0; r < hiddenDim; r++) {
      for (let c = 0; c < hiddenDim; c++) {
        G[r][c] += h[r] * h[c];
      }
    }
  }

  // Eigen decomposition using mathjs
  const gMatrix = math.matrix(Array.from(G).map(row => Array.from(row)));
  const { eigenvectors } = math.eigs(gMatrix);

  // mathjs eigs returns eigenvectors as an array of { value, vector }
  const eigenPairs = eigenvectors.map((p: any) => ({
    value: typeof p.value === 'number' ? p.value : p.value.toNumber(),
    vector: p.vector.toArray()
  }));

  eigenPairs.sort((a: any, b: any) => b.value - a.value);

  const topKValues = eigenPairs.slice(0, k).map((p: any) => p.value);
  const topKVectors = eigenPairs.slice(0, k).map((p: any) => p.vector);

  // Transpose topKVectors to get V_k (hiddenDim x k)
  const Vk = Array.from({ length: hiddenDim }, (_, r) => 
    Array.from({ length: k }, (_, c) => topKVectors[c][r])
  );

  return { evals: topKValues, Vk };
}
