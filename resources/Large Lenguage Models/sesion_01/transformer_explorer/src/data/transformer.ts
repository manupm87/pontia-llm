export type LayerId =
  | 'overview'
  | 'source-tokens'
  | 'positional-embedding'
  | 'encoder'
  | 'global-self-attention'
  | 'feed-forward'
  | 'target-shift'
  | 'decoder'
  | 'causal-self-attention'
  | 'cross-attention'
  | 'dense-output'
  | 'autoregressive-loop';

export type LayerGroup = 'input' | 'encoder' | 'decoder' | 'output' | 'inference';

export interface Subcomponent {
  name: string;
  role: string;
  shape: string;
}

export interface LayerDetail {
  id: LayerId;
  title: string;
  shortName: string;
  group: LayerGroup;
  subtitle: string;
  keras: string;
  input: string;
  operation: string;
  output: string;
  meaning: string;
  shape: string;
  highlight: string;
  sublayers?: Subcomponent[];
}

export interface DiagramNode {
  id: string;
  layerId: LayerId;
  title: string;
  subtitle: string;
  x: number;
  y: number;
  w: number;
  h: number;
  group: LayerGroup;
}

export interface DiagramEdge {
  from: string;
  to: string;
  label?: string;
  emphasized?: boolean;
}

export interface AttentionVisual {
  title: string;
  note: string;
  querySource: string;
  keySource: string;
  valueSource: string;
  rowsLabel: string;
  columnsLabel: string;
  rows: string[];
  columns: string[];
  matrix: number[][];
}

export const example = {
  sourceSentence: 'este é um problema que temos que resolver.',
  expectedTranslation: 'this is a problem we have to solve .',
  sourceTokens: ['[START]', 'este', 'é', 'um', 'problema', 'que', 'temos', 'que', 'resolver', '.', '[END]'],
  decoderInput: ['[START]', 'this', 'is', 'a', 'problem', 'we', 'have', 'to', 'solve', '.'],
  targetLabels: ['this', 'is', 'a', 'problem', 'we', 'have', 'to', 'solve', '.', '[END]'],
};

export const layerDetails: Record<LayerId, LayerDetail> = {
  overview: {
    id: 'overview',
    title: 'Flujo del Transformer',
    shortName: 'Overview',
    group: 'inference',
    subtitle: 'Encoder-decoder para traducción portugués-inglés.',
    keras: 'class Transformer(tf.keras.Model)',
    input: '`context` recibe IDs portugueses y `x` recibe IDs ingleses desplazados.',
    operation: 'El encoder produce contexto contextualizado; el decoder lo consulta mientras genera la secuencia inglesa.',
    output: 'Logits con forma `(batch, target_len, target_vocab_size)`.',
    meaning: 'El modelo aprende a convertir una frase portuguesa en una distribución de probabilidad sobre el vocabulario inglés en cada paso.',
    shape: '(pt_ids, en_ids) -> logits',
    highlight: 'La traducción se entrena con teacher forcing y se usa en inferencia de forma autoregresiva.',
  },
  'source-tokens': {
    id: 'source-tokens',
    title: 'Entrada portuguesa tokenizada',
    shortName: 'Tokens PT',
    group: 'input',
    subtitle: 'La frase se convierte en IDs enteros antes de entrar al encoder.',
    keras: 'tokenizers.pt.tokenize(...)',
    input: '`este é um problema que temos que resolver.`',
    operation: 'El tokenizador WordPiece añade tokens especiales, recorta a `MAX_TOKENS` y rellena con ceros cuando hace falta.',
    output: 'Tensor denso con forma `(batch, context_len)`.',
    meaning: 'El Transformer no ve texto crudo: ve índices de vocabulario que representan piezas de palabra en portugués.',
    shape: '(1, 11)',
    highlight: 'El token `0` queda reservado para padding y se enmascara gracias a `mask_zero=True`.',
  },
  'positional-embedding': {
    id: 'positional-embedding',
    title: 'Embedding + codificación posicional',
    shortName: 'PositionalEmbedding',
    group: 'input',
    subtitle: 'Cada token se proyecta a `d_model` y recibe información de posición.',
    keras: 'class PositionalEmbedding(tf.keras.layers.Layer)',
    input: 'IDs de tokens con forma `(batch, seq_len)`.',
    operation: 'Busca embeddings entrenables, los escala por `sqrt(d_model)` y suma senos/cosenos posicionales fijos.',
    output: 'Representaciones con forma `(batch, seq_len, d_model)`.',
    meaning: 'Ahora `problema` tiene un vector semántico y además sabe que aparece después de `um` y antes de `que`.',
    shape: '(1, 11) -> (1, 11, 128)',
    highlight: 'Sin esta suma posicional, la autoatención no distinguiría el orden de los tokens.',
    sublayers: [
      { name: 'Embedding', role: 'Tabla entrenable que asigna un vector a cada ID de token.', shape: '(vocab, 128)' },
      { name: 'Escalado', role: 'Multiplica por `sqrt(d_model)` para equilibrar magnitudes.', shape: '(batch, seq, 128)' },
      { name: 'Positional encoding', role: 'Suma senos/cosenos fijos que codifican posición absoluta.', shape: '(2048, 128)' },
      { name: 'Mask', role: '`mask_zero=True` propaga la máscara de padding.', shape: '(batch, seq)' },
    ],
  },
  encoder: {
    id: 'encoder',
    title: 'Encoder x4',
    shortName: 'Encoder',
    group: 'encoder',
    subtitle: 'Cuatro capas transforman la frase portuguesa en contexto.',
    keras: 'class Encoder / class EncoderLayer',
    input: 'Embeddings portugueses con forma `(batch, context_len, d_model)`.',
    operation: 'Cada capa aplica autoatención global, residual, normalización y una red feed-forward punto a punto.',
    output: 'Contexto con forma `(batch, context_len, d_model)`.',
    meaning: 'Cada token portugués se reescribe usando información de toda la frase, por ejemplo `resolver` queda ligado a `problema`.',
    shape: '(1, 11, 128) -> (1, 11, 128)',
    highlight: 'La longitud no cambia; cambia la calidad contextual de cada vector.',
    sublayers: [
      { name: 'PositionalEmbedding', role: 'Convierte IDs portugueses en vectores ordenados.', shape: '(1, 11, 128)' },
      { name: 'Dropout', role: 'Regulariza las representaciones durante entrenamiento.', shape: '(1, 11, 128)' },
      { name: 'EncoderLayer x4', role: 'Repite autoatención global y feed-forward cuatro veces.', shape: '(1, 11, 128)' },
    ],
  },
  'global-self-attention': {
    id: 'global-self-attention',
    title: 'Autoatención global',
    shortName: 'GlobalSelfAttention',
    group: 'encoder',
    subtitle: 'Cada token portugués atiende a todos los demás tokens portugueses.',
    keras: 'class GlobalSelfAttention(BaseAttention)',
    input: '`query`, `key` y `value` son el mismo tensor del encoder.',
    operation: 'MultiHeadAttention calcula compatibilidades token-token en 8 cabezas y mezcla valores relevantes.',
    output: 'Tensor contextualizado con residual y LayerNorm.',
    meaning: 'El encoder puede relacionar `problema` con `resolver` aunque no estén contiguos.',
    shape: '(1, 11, 128) -> (1, 11, 128)',
    highlight: 'Es bidireccional porque la frase de entrada ya está completa.',
    sublayers: [
      { name: 'Query', role: 'Qué pregunta hace cada token portugués al resto de la frase.', shape: '(batch, heads, 11, depth)' },
      { name: 'Key', role: 'Qué información ofrece cada token portugués para ser encontrado.', shape: '(batch, heads, 11, depth)' },
      { name: 'Value', role: 'Contenido que se mezcla cuando un token recibe atención.', shape: '(batch, heads, 11, depth)' },
      { name: 'Add + LayerNorm', role: 'Suma residual y estabiliza la escala de activaciones.', shape: '(1, 11, 128)' },
    ],
  },
  'feed-forward': {
    id: 'feed-forward',
    title: 'Red feed-forward',
    shortName: 'FeedForward',
    group: 'encoder',
    subtitle: 'Transformación no lineal aplicada independientemente a cada posición.',
    keras: 'class FeedForward(tf.keras.layers.Layer)',
    input: 'Vectores de dimensión `d_model = 128`.',
    operation: 'Dense(512, relu) expande capacidad, Dense(128) vuelve a `d_model`, dropout, residual y normalización.',
    output: 'Tensor con la misma forma de entrada.',
    meaning: 'Después de mezclar información entre tokens, cada posición refina su representación con una MLP compartida.',
    shape: '(1, seq_len, 128) -> (1, seq_len, 128)',
    highlight: 'La atención mezcla posiciones; la feed-forward aumenta la expresividad de cada posición.',
    sublayers: [
      { name: 'Dense + ReLU', role: 'Expande cada vector de 128 a 512 dimensiones.', shape: '128 -> 512' },
      { name: 'Dense', role: 'Devuelve la representación a `d_model`.', shape: '512 -> 128' },
      { name: 'Dropout', role: 'Evita dependencia excesiva de activaciones concretas.', shape: '(batch, seq, 128)' },
      { name: 'Add + LayerNorm', role: 'Conserva la ruta residual y normaliza.', shape: '(batch, seq, 128)' },
    ],
  },
  'target-shift': {
    id: 'target-shift',
    title: 'Entrada inglesa desplazada',
    shortName: 'Teacher forcing',
    group: 'input',
    subtitle: 'Durante entrenamiento el decoder ve la traducción parcial correcta.',
    keras: 'prepare_batch(...)',
    input: 'Traducción inglesa tokenizada con `[START]` y `[END]`.',
    operation: '`en_inputs = en[:, :-1]` entra al decoder y `en_labels = en[:, 1:]` se usa como etiqueta.',
    output: 'Pares entrada-etiqueta desplazados un token.',
    meaning: 'Con entrada `[START] this is`, el modelo aprende a predecir `this is a` paso a paso.',
    shape: 'labels[t] = inputs[t + 1]',
    highlight: 'Esto enseña generación autoregresiva sin esperar a que el modelo se equivoque durante el entrenamiento.',
  },
  decoder: {
    id: 'decoder',
    title: 'Decoder x4',
    shortName: 'Decoder',
    group: 'decoder',
    subtitle: 'Genera representaciones inglesas condicionadas por el contexto portugués.',
    keras: 'class Decoder / class DecoderLayer',
    input: 'Tokens ingleses previos y salida completa del encoder.',
    operation: 'Cada capa aplica autoatención causal, atención cruzada al encoder y feed-forward.',
    output: 'Vectores ingleses con forma `(batch, target_len, d_model)`.',
    meaning: 'Para predecir `problem`, el decoder usa lo que ya produjo en inglés y mira el contexto portugués `problema`.',
    shape: '(1, 10, 128) + context -> (1, 10, 128)',
    highlight: 'El decoder combina memoria de salida y alineamiento con la entrada.',
    sublayers: [
      { name: 'PositionalEmbedding', role: 'Convierte los tokens ingleses ya generados en vectores ordenados.', shape: '(1, 10, 128)' },
      { name: 'CausalSelfAttention', role: 'Lee solo tokens ingleses ya disponibles.', shape: '(1, 10, 128)' },
      { name: 'CrossAttention', role: 'Consulta el contexto portugués generado por el encoder.', shape: '(1, 10, 128)' },
      { name: 'FeedForward', role: 'Refina cada posición antes de la proyección final.', shape: '(1, 10, 128)' },
    ],
  },
  'causal-self-attention': {
    id: 'causal-self-attention',
    title: 'Autoatención causal',
    shortName: 'CausalSelfAttention',
    group: 'decoder',
    subtitle: 'El token actual solo puede mirar al pasado de la traducción.',
    keras: 'class CausalSelfAttention(BaseAttention)',
    input: 'Embeddings ingleses desplazados.',
    operation: 'MultiHeadAttention usa `use_causal_mask=True` para ocultar posiciones futuras.',
    output: 'Representación de los tokens ingleses ya generados hasta cada posición.',
    meaning: 'Cuando el modelo predice `problem`, no puede mirar la etiqueta futura `we`.',
    shape: '(1, target_len, 128) -> (1, target_len, 128)',
    highlight: 'La máscara causal hace que entrenamiento e inferencia tengan la misma restricción temporal.',
    sublayers: [
      { name: 'Query', role: 'Token inglés que pregunta por los tokens ingleses ya generados.', shape: '(batch, heads, target_len, depth)' },
      { name: 'Key', role: 'Tokens ingleses previos contra los que se compara.', shape: '(batch, heads, target_len, depth)' },
      { name: 'Value', role: 'Contenido de los tokens ingleses ya generados que se puede copiar o combinar.', shape: '(batch, heads, target_len, depth)' },
      { name: 'Causal mask', role: 'Anula la atención a posiciones futuras.', shape: '(target_len, target_len)' },
      { name: 'Add + LayerNorm', role: 'Residual y normalización tras mezclar los tokens ingleses ya generados.', shape: '(1, target_len, 128)' },
    ],
  },
  'cross-attention': {
    id: 'cross-attention',
    title: 'Atención cruzada',
    shortName: 'CrossAttention',
    group: 'decoder',
    subtitle: 'El decoder consulta la frase portuguesa codificada.',
    keras: 'class CrossAttention(BaseAttention)',
    input: '`query` viene del decoder; `key` y `value` vienen del encoder.',
    operation: 'Calcula qué tokens portugueses son relevantes para cada token inglés y guarda `last_attn_scores`.',
    output: 'Vectores ingleses enriquecidos con contexto portugués.',
    meaning: '`problem` debería concentrarse en `problema`, y `solve` en `resolver`.',
    shape: '(1, target_len, 128) x (1, context_len, 128)',
    highlight: 'Es el puente conceptual entre comprender portugués y producir inglés.',
    sublayers: [
      { name: 'Query', role: 'Viene del decoder: qué necesita el token inglés actual.', shape: '(batch, heads, target_len, depth)' },
      { name: 'Key', role: 'Viene del encoder: cómo se indexan los tokens portugueses.', shape: '(batch, heads, context_len, depth)' },
      { name: 'Value', role: 'Viene del encoder: información portuguesa que se transfiere.', shape: '(batch, heads, context_len, depth)' },
      { name: 'Attention scores', role: '`last_attn_scores` permite visualizar alineamientos.', shape: '(batch, heads, target_len, context_len)' },
      { name: 'Add + LayerNorm', role: 'Integra el contexto portugués en el stream inglés.', shape: '(1, target_len, 128)' },
    ],
  },
  'dense-output': {
    id: 'dense-output',
    title: 'Capa Dense final',
    shortName: 'Dense vocab',
    group: 'output',
    subtitle: 'Convierte cada vector del decoder en logits sobre el vocabulario inglés.',
    keras: 'tf.keras.layers.Dense(target_vocab_size)',
    input: 'Salida del decoder con forma `(batch, target_len, d_model)`.',
    operation: 'Una proyección lineal produce una puntuación por token del vocabulario inglés.',
    output: 'Logits con forma `(batch, target_len, 7010)`.',
    meaning: 'En cada posición decide si el siguiente token más probable es `this`, `problem`, `solve`, etc.',
    shape: '(1, 10, 128) -> (1, 10, 7010)',
    highlight: 'La pérdida compara estos logits con `en_labels`, ignorando padding mediante máscara.',
    sublayers: [
      { name: 'Dense', role: 'Proyecta cada vector del decoder a una puntuación por token inglés.', shape: '128 -> 7010' },
      { name: 'Masked loss', role: 'Calcula entropía cruzada sin contar padding.', shape: '(batch, target_len)' },
      { name: 'Masked accuracy', role: 'Evalúa aciertos reales sobre tokens no padding.', shape: 'scalar' },
    ],
  },
  'autoregressive-loop': {
    id: 'autoregressive-loop',
    title: 'Bucle autoregresivo',
    shortName: 'Inferencia',
    group: 'inference',
    subtitle: 'En producción el modelo genera un token cada vez.',
    keras: 'class Translator(tf.Module)',
    input: 'Frase portuguesa y salida inicial `[START]`.',
    operation: 'Predice el último token, lo concatena a la salida y repite hasta `[END]` o `MAX_TOKENS`.',
    output: 'Texto detokenizado en inglés.',
    meaning: 'La traducción se construye como `this` -> `this is` -> `this is a` hasta completar la frase.',
    shape: 'loop <= 128 pasos',
    highlight: 'La app muestra el proceso de forma didáctica, sin ejecutar el SavedModel.',
    sublayers: [
      { name: '[START]', role: 'Inicializa la salida inglesa.', shape: '(1, 1)' },
      { name: 'Transformer call', role: 'Recalcula logits con los tokens ingleses acumulados.', shape: '(1, t, 7010)' },
      { name: 'argmax', role: 'Selecciona el ID más probable del último paso.', shape: '(1, 1)' },
      { name: 'append', role: 'Añade el token y continúa hasta `[END]`.', shape: 't -> t + 1' },
    ],
  },
};

export const diagramNodes: DiagramNode[] = [
  { id: 'pt', layerId: 'source-tokens', title: 'PT tokens', subtitle: '(1, 11)', x: 60, y: 230, w: 220, h: 90, group: 'input' },
  { id: 'pt-embed', layerId: 'positional-embedding', title: 'PositionalEmbedding', subtitle: '(1, 11, 128)', x: 340, y: 230, w: 260, h: 90, group: 'input' },
  { id: 'encoder', layerId: 'encoder', title: 'Encoder x4', subtitle: 'Global self-attn + feed-forward', x: 680, y: 95, w: 380, h: 345, group: 'encoder' },
  { id: 'gsa', layerId: 'global-self-attention', title: 'Global self-attn', subtitle: '8 heads', x: 725, y: 235, w: 290, h: 78, group: 'encoder' },
  { id: 'ffn-enc', layerId: 'feed-forward', title: 'Feed-forward', subtitle: '128 -> 512 -> 128', x: 725, y: 340, w: 290, h: 78, group: 'encoder' },
  { id: 'context', layerId: 'encoder', title: 'Contexto encoder', subtitle: '(1, 11, 128)', x: 1170, y: 230, w: 260, h: 90, group: 'encoder' },
  { id: 'en', layerId: 'target-shift', title: 'EN shifted input', subtitle: '[START] this is ...', x: 60, y: 735, w: 240, h: 96, group: 'input' },
  { id: 'en-embed', layerId: 'positional-embedding', title: 'PositionalEmbedding', subtitle: '(1, 10, 128)', x: 370, y: 738, w: 260, h: 90, group: 'input' },
  { id: 'decoder', layerId: 'decoder', title: 'Decoder x4', subtitle: 'Causal self-attn + cross-attn + feed-forward', x: 680, y: 590, w: 410, h: 445, group: 'decoder' },
  { id: 'csa', layerId: 'causal-self-attention', title: 'Causal self-attn', subtitle: 'masked', x: 730, y: 740, w: 310, h: 76, group: 'decoder' },
  { id: 'cross', layerId: 'cross-attention', title: 'Cross-attention', subtitle: 'decoder -> encoder', x: 730, y: 845, w: 310, h: 76, group: 'decoder' },
  { id: 'ffn-dec', layerId: 'feed-forward', title: 'Feed-forward', subtitle: 'per position', x: 730, y: 950, w: 310, h: 76, group: 'decoder' },
  { id: 'dense', layerId: 'dense-output', title: 'Dense vocab EN', subtitle: '(1, 10, 7010)', x: 1240, y: 735, w: 240, h: 90, group: 'output' },
  { id: 'loop', layerId: 'autoregressive-loop', title: 'Autoregressive loop', subtitle: 'append argmax', x: 1540, y: 735, w: 220, h: 90, group: 'inference' },
];

export const diagramEdges: DiagramEdge[] = [
  { from: 'pt', to: 'pt-embed', label: 'IDs' },
  { from: 'pt-embed', to: 'encoder' },
  { from: 'encoder', to: 'context', label: 'context' },
  { from: 'en', to: 'en-embed', label: 'shifted' },
  { from: 'en-embed', to: 'decoder' },
  { from: 'context', to: 'cross', label: 'K,V', emphasized: true },
  { from: 'decoder', to: 'dense' },
  { from: 'dense', to: 'loop', label: 'argmax' },
  { from: 'loop', to: 'en', label: 'next token', emphasized: true },
];

const crossPeakByRow: Record<string, string[]> = {
  this: ['este'],
  is: ['é'],
  a: ['um'],
  problem: ['problema'],
  we: ['temos'],
  have: ['temos', 'que'],
  to: ['que'],
  solve: ['resolver'],
  '.': ['.'],
  '[END]': ['[END]'],
};

const crossMatrix = example.targetLabels.map((row) => {
  const peaks = crossPeakByRow[row] ?? [];
  return example.sourceTokens.map((column) => {
    if (peaks.includes(column)) return 0.92;
    if (column === '[START]' || column === '[END]') return row === column ? 0.72 : 0.06;
    if (column === 'que' && (row === 'have' || row === 'to')) return 0.62;
    return 0.14;
  });
});

const sourceRelation: Record<string, string[]> = {
  este: ['é'],
  é: ['este', 'problema'],
  um: ['problema'],
  problema: ['resolver', 'um'],
  que: ['temos', 'resolver'],
  temos: ['que', 'resolver'],
  resolver: ['problema', 'temos'],
};

const globalSelfMatrix = example.sourceTokens.map((row) =>
  example.sourceTokens.map((column) => {
    if (row === column) return 0.7;
    if ((sourceRelation[row] ?? []).includes(column)) return 0.88;
    if (row === '[START]' || row === '[END]' || column === '[START]' || column === '[END]') return 0.05;
    return 0.16;
  }),
);

const causalMatrix = example.decoderInput.map((row, rowIndex) =>
  example.decoderInput.map((column, columnIndex) => {
    if (columnIndex > rowIndex) return 0.02;
    if (columnIndex === rowIndex) return 0.74;
    if (row === 'problem' && column === 'a') return 0.82;
    if (row === 'solve' && (column === 'to' || column === 'have')) return 0.72;
    if (row === 'we' && column === 'problem') return 0.58;
    return Math.max(0.12, 0.38 - (rowIndex - columnIndex) * 0.04);
  }),
);

export const attentionVisuals: Partial<Record<LayerId, AttentionVisual>> = {
  'global-self-attention': {
    title: 'Autoatención global: Q, K y V salen del encoder',
    note: 'Visualización didáctica de una cabeza: todos los tokens portugueses pueden atender a todos los tokens portugueses porque la frase de entrada ya está completa.',
    querySource: 'Q = tokens portugueses contextualizándose',
    keySource: 'K = los mismos tokens portugueses como índice de búsqueda',
    valueSource: 'V = los mismos tokens portugueses como contenido mezclable',
    rowsLabel: 'Q / token PT',
    columnsLabel: 'K,V / token PT',
    rows: example.sourceTokens,
    columns: example.sourceTokens,
    matrix: globalSelfMatrix,
  },
  'causal-self-attention': {
    title: 'Autoatención causal: Q, K y V se proyectan desde los tokens ingleses ya generados',
    note: 'Visualización didáctica de una cabeza: Q, K y V nacen de las mismas posiciones inglesas disponibles; la máscara causal limita qué keys/values puede consultar cada query.',
    querySource: 'Q = qué busca cada posición inglesa ya generada',
    keySource: 'K = cómo se identifica cada posición inglesa ya generada',
    valueSource: 'V = información que aporta cada posición inglesa ya generada',
    rowsLabel: 'Q / token EN',
    columnsLabel: 'K,V / tokens EN generados',
    rows: example.decoderInput,
    columns: example.decoderInput,
    matrix: causalMatrix,
  },
  'cross-attention': {
    title: 'Atención cruzada: Q del decoder, K/V del encoder',
    note: 'Visualización didáctica de una cabeza: ilustra alineamientos esperables como `problem` → `problema` y `solve` → `resolver`; no son pesos reales extraídos del modelo.',
    querySource: 'Q = tokens ingleses del decoder',
    keySource: 'K = tokens portugueses codificados por el encoder',
    valueSource: 'V = contexto portugués que se transfiere al decoder',
    rowsLabel: 'Q / token EN',
    columnsLabel: 'K,V / token PT',
    rows: example.targetLabels,
    columns: example.sourceTokens,
    matrix: crossMatrix,
  },
};
