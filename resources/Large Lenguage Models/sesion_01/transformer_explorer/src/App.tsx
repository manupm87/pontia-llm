import { Fragment, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  ArrowRight,
  BrainCircuit,
  Eye,
  Layers,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
  Sigma,
} from 'lucide-react';
import {
  attentionVisuals,
  diagramEdges,
  diagramNodes,
  example,
  layerDetails,
  type AttentionVisual,
  type DiagramEdge,
  type DiagramNode,
  type LayerDetail,
  type LayerId,
  type Subcomponent,
} from './data/transformer';

interface TourStep {
  layerId: LayerId;
  nodeId?: string;
  generatedTokens?: number;
}

const tour: TourStep[] = [
  { layerId: 'source-tokens', nodeId: 'pt' },
  { layerId: 'positional-embedding', nodeId: 'pt-embed' },
  { layerId: 'encoder', nodeId: 'encoder' },
  { layerId: 'global-self-attention', nodeId: 'gsa' },
  { layerId: 'feed-forward', nodeId: 'ffn-enc' },
  { layerId: 'encoder', nodeId: 'context' },
  { layerId: 'target-shift', nodeId: 'en', generatedTokens: 0 },
  { layerId: 'positional-embedding', nodeId: 'en-embed', generatedTokens: 0 },
  { layerId: 'decoder', nodeId: 'decoder', generatedTokens: 0 },
  { layerId: 'causal-self-attention', nodeId: 'csa', generatedTokens: 0 },
  { layerId: 'cross-attention', nodeId: 'cross', generatedTokens: 0 },
  { layerId: 'feed-forward', nodeId: 'ffn-dec', generatedTokens: 0 },
  { layerId: 'dense-output', nodeId: 'dense', generatedTokens: 0 },
  ...example.targetLabels.map((_, index) => ({
    layerId: 'autoregressive-loop' as LayerId,
    nodeId: 'loop',
    generatedTokens: index + 1,
  })),
];

const groupLabels = {
  input: 'Datos',
  encoder: 'Encoder',
  decoder: 'Decoder',
  output: 'Salida',
  inference: 'Inferencia',
};

const DIAGRAM_WIDTH = 1800;
const DIAGRAM_HEIGHT = 1220;

function App() {
  const [selectedId, setSelectedId] = useState<LayerId>('overview');
  const [playing, setPlaying] = useState(false);
  const [tourIndex, setTourIndex] = useState(0);

  const activeTourStep = playing ? tour[tourIndex] : undefined;
  const selected = layerDetails[selectedId];
  const selectedAttention = attentionVisuals[selectedId];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setTourIndex((current) => {
        const next = (current + 1) % tour.length;
        setSelectedId(tour[next].layerId);
        return next;
      });
    }, 1700);

    return () => window.clearInterval(timer);
  }, [playing]);

  const activeNodeIds = useMemo(() => {
    if (activeTourStep?.nodeId) {
      return new Set([activeTourStep.nodeId]);
    }

    return new Set(diagramNodes.filter((node) => node.layerId === selectedId).map((node) => node.id));
  }, [activeTourStep?.nodeId, selectedId]);

  const visibleTargetTokens = activeTourStep?.generatedTokens;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <BrainCircuit size={24} />
          </div>
          <div>
            <p className="eyebrow">Pontia IA Generativa · Sesión 01</p>
            <h1>Transformer Explorer</h1>
          </div>
        </div>

        <div className="topbar-actions">
          <button
            className="icon-button text-button"
            type="button"
            onClick={() => {
              const next = !playing;
              setPlaying(next);
              if (next) {
                setSelectedId(tour[0].layerId);
                setTourIndex(0);
              }
            }}
            title={playing ? 'Pausar recorrido' : 'Reproducir recorrido'}
          >
            {playing ? <Pause size={18} /> : <Play size={18} />}
            <span>{playing ? 'Pausar' : 'Recorrido'}</span>
          </button>
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              setPlaying(false);
              setSelectedId('overview');
            }}
            title="Ver arquitectura completa"
          >
            <Maximize2 size={18} />
          </button>
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              setPlaying(false);
              setTourIndex(0);
              setSelectedId(tour[0].layerId);
            }}
            title="Reiniciar recorrido"
          >
            <RotateCcw size={18} />
          </button>
        </div>
      </header>

      <section className="summary-band">
        <h2>Traducción portugués → inglés</h2>
        <TokenStrip label="ENTRADA DE EJEMPLO EN PORTUGUÉS" tokens={example.sourceTokens} tone="source" />
        <TokenStrip
          label="SALIDA ESPERADA EN INGLÉS"
          tokens={example.targetLabels}
          tone="target"
          visibleCount={visibleTargetTokens}
        />
      </section>

      <section className="workspace">
        <div className="diagram-panel">
          <div className="panel-heading">
            <div>
              <h2>{selected.title}</h2>
            </div>
            <span className={`group-pill ${selected.group}`}>{groupLabels[selected.group]}</span>
          </div>

          <TransformerDiagram
            selectedId={selectedId}
            activeNodeIds={activeNodeIds}
            onSelect={(id) => {
              setPlaying(false);
              setSelectedId((current) => (current === id ? 'overview' : id));
              const index = tour.findIndex((step) => step.layerId === id);
              if (index >= 0) setTourIndex(index);
            }}
          />
        </div>

        <aside className="detail-panel">
          <div className="detail-header">
            <div className="detail-icon" aria-hidden="true">
              <Layers size={22} />
            </div>
            <div>
              <h2>{selected.title}</h2>
            </div>
          </div>

          <p className="subtitle">{selected.subtitle}</p>

          <div className="shape-badge">
            <Sigma size={18} />
            <span>{selected.shape}</span>
          </div>

          {selected.sublayers && <ComponentBreakdown layerTitle={selected.title} items={selected.sublayers} />}

          <DetailSection icon={<ArrowRight size={16} />} title="Qué entra" text={selected.input} />
          <DetailSection icon={<BrainCircuit size={16} />} title="Qué ocurre" text={selected.operation} />
          <DetailSection icon={<Eye size={16} />} title="Qué significa aquí" text={selected.meaning} />

          <div className="callout">{selected.highlight}</div>
        </aside>
      </section>

      {selectedAttention && <AttentionDeepDive visual={selectedAttention} />}
    </main>
  );
}

interface DiagramProps {
  selectedId: LayerId;
  activeNodeIds: Set<string>;
  onSelect: (id: LayerId) => void;
}

function TransformerDiagram({ selectedId, activeNodeIds, onSelect }: DiagramProps) {
  const selectedDetail = layerDetails[selectedId];

  const viewBox = useMemo(() => {
    if (selectedId === 'overview') return `0 0 ${DIAGRAM_WIDTH} ${DIAGRAM_HEIGHT}`;

    const selectedNodes = diagramNodes.filter((node) => node.layerId === selectedId);
    if (!selectedNodes.length) return `0 0 ${DIAGRAM_WIDTH} ${DIAGRAM_HEIGHT}`;

    const minX = Math.min(...selectedNodes.map((node) => node.x));
    const minY = Math.min(...selectedNodes.map((node) => node.y));
    const maxX = Math.max(...selectedNodes.map((node) => node.x + node.w));
    const maxY = Math.max(...selectedNodes.map((node) => node.y + node.h));

    const hasExplodedFlow = Boolean(selectedDetail.sublayers?.length);
    const padX = hasExplodedFlow ? 520 : selectedId === 'cross-attention' ? 280 : 170;
    const padY = hasExplodedFlow ? 320 : selectedId === 'encoder' || selectedId === 'decoder' ? 105 : 130;
    return `${Math.max(0, minX - padX)} ${Math.max(0, minY - padY)} ${Math.min(DIAGRAM_WIDTH, maxX - minX + padX * 2)} ${Math.min(DIAGRAM_HEIGHT, maxY - minY + padY * 2)}`;
  }, [selectedDetail.sublayers, selectedId]);

  const explodedNode = useMemo(() => getExplodedNode(selectedId, activeNodeIds), [activeNodeIds, selectedId]);

  return (
    <div className="diagram-viewport">
      <svg viewBox={viewBox} role="img" aria-label="Diagrama interactivo de arquitectura Transformer">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L8,3 z" />
          </marker>
        </defs>

        <rect className="lane encoder-lane" x="645" y="50" width="850" height="430" rx="8" />
        <text className="lane-title" x="675" y="82">
          Encoder: comprensión de la frase portuguesa
        </text>
        <rect className="lane decoder-lane" x="645" y="510" width="1080" height="550" rx="8" />
        <text className="lane-title" x="675" y="542">
          Decoder: generación condicionada en inglés
        </text>

        {diagramEdges.map((edge) => (
          <Edge key={`${edge.from}-${edge.to}`} edge={edge} activeNodeIds={activeNodeIds} />
        ))}

        {diagramNodes.map((node) => (
          <NodeBlock
            key={node.id}
            node={node}
            selected={activeNodeIds.has(node.id)}
            dimmed={selectedId !== 'overview' && !activeNodeIds.has(node.id)}
            onClick={() => onSelect(node.layerId)}
          />
        ))}

        {selectedId !== 'overview' && selectedDetail.sublayers && explodedNode && (
          <ExplodedSubflow node={explodedNode} detail={selectedDetail} />
        )}
      </svg>
    </div>
  );
}

function Edge({ edge, activeNodeIds }: { edge: DiagramEdge; activeNodeIds: Set<string> }) {
  const from = diagramNodes.find((node) => node.id === edge.from);
  const to = diagramNodes.find((node) => node.id === edge.to);
  if (!from || !to) return null;

  const start = { x: from.x + from.w, y: from.y + from.h / 2 };
  const end = { x: to.x, y: to.y + to.h / 2 };
  const active = activeNodeIds.has(edge.from) || activeNodeIds.has(edge.to) || edge.emphasized;

  let path =
    edge.from === 'loop'
      ? `M ${from.x + from.w / 2} ${from.y + from.h}
         C ${from.x + 140} 1138, ${from.x - 70} 1165, ${from.x - 330} 1165
         L ${to.x + to.w / 2} 1165
         L ${to.x + to.w / 2} ${to.y + to.h + 10}`
      : `M ${start.x} ${start.y} C ${start.x + 95} ${start.y}, ${end.x - 95} ${end.y}, ${end.x} ${end.y}`;

  if (edge.from === 'context' && edge.to === 'cross') {
    const crossEnd = { x: to.x + to.w, y: to.y + to.h / 2 };
    path = `M ${from.x + from.w / 2} ${from.y + from.h} C ${from.x + 160} 515, ${crossEnd.x + 180} 735, ${crossEnd.x} ${crossEnd.y}`;
  }

  let labelX = edge.from === 'loop' ? 560 : (start.x + end.x) / 2;
  let labelY = edge.from === 'loop' ? 1144 : (start.y + end.y) / 2 - 12;
  if (edge.from === 'context' && edge.to === 'cross') {
    labelX = 1255;
    labelY = 582;
  }

  return (
    <g className={`edge ${active ? 'active' : ''}`}>
      <path d={path} markerEnd="url(#arrow)" />
      {edge.label && (
        <text x={labelX} y={labelY}>
          {edge.label}
        </text>
      )}
    </g>
  );
}

function NodeBlock({
  node,
  selected,
  dimmed,
  onClick,
}: {
  node: DiagramNode;
  selected: boolean;
  dimmed: boolean;
  onClick: () => void;
}) {
  const stackCount = getStackCount(node.id);

  return (
    <g
      className={`node node-${node.id} ${node.group} ${selected ? 'selected' : ''} ${dimmed ? 'dimmed' : ''}`}
      onClick={onClick}
      tabIndex={0}
      role="button"
      aria-label={`Abrir detalle de ${node.title}`}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') onClick();
      }}
    >
      {Array.from({ length: stackCount }).map((_, index) => {
        const offset = (stackCount - index) * 7;
        return (
          <rect
            className="stack-plate"
            key={`${node.id}-stack-${index}`}
            x={node.x + offset}
            y={node.y + offset}
            width={node.w}
            height={node.h}
            rx="8"
          />
        );
      })}
      <rect x={node.x} y={node.y} width={node.w} height={node.h} rx="8" />
      <foreignObject x={node.x + 14} y={node.y + 12} width={node.w - 28} height={node.h - 24}>
        <div className="node-copy">
          <strong>{node.title}</strong>
          <span>{node.subtitle}</span>
        </div>
      </foreignObject>
    </g>
  );
}

function getStackCount(nodeId: string) {
  if (nodeId === 'encoder' || nodeId === 'decoder') return 3;
  if (['gsa', 'csa', 'cross'].includes(nodeId)) return 2;
  return 0;
}

function getExplodedNode(selectedId: LayerId, activeNodeIds: Set<string>) {
  const focusedNode = diagramNodes.find((node) => activeNodeIds.has(node.id) && node.layerId === selectedId);
  if (focusedNode) return focusedNode;

  const preferredByLayer: Partial<Record<LayerId, string>> = {
    'positional-embedding': 'pt-embed',
    encoder: 'encoder',
    'global-self-attention': 'gsa',
    'feed-forward': 'ffn-enc',
    decoder: 'decoder',
    'causal-self-attention': 'csa',
    'cross-attention': 'cross',
    'dense-output': 'dense',
    'autoregressive-loop': 'loop',
  };

  const preferredId = preferredByLayer[selectedId];
  return diagramNodes.find((node) => node.id === preferredId) ?? diagramNodes.find((node) => node.layerId === selectedId);
}

function ExplodedSubflow({ node, detail }: { node: DiagramNode; detail: LayerDetail }) {
  const items = detail.sublayers ?? [];
  const boxW = items.length >= 5 ? 138 : 154;
  const boxH = 84;
  const gap = 18;
  const flowW = items.length * boxW + Math.max(0, items.length - 1) * gap;
  const x = Math.max(34, Math.min(DIAGRAM_WIDTH - flowW - 34, node.x + node.w / 2 - flowW / 2));
  const belowY = node.y + node.h + 48;
  const y = belowY + boxH > DIAGRAM_HEIGHT - 24 ? Math.max(34, node.y - boxH - 58) : belowY;
  const isAttention = ['global-self-attention', 'causal-self-attention', 'cross-attention'].includes(detail.id);

  return (
    <g className={`subflow ${isAttention ? 'attention-subflow' : ''}`} aria-label={`Subcapas de ${detail.title}`}>
      <text className="subflow-title" x={x} y={y - 18}>
        Subcapas dentro de {detail.title}
      </text>
      {items.map((item, index) => {
        const itemX = x + index * (boxW + gap);
        const isQkv = ['Query', 'Key', 'Value'].includes(item.name);
        return (
          <g className={`subflow-node ${isQkv ? `qkv-${item.name.toLowerCase()}` : ''}`} key={`${detail.id}-${item.name}`}>
            {index > 0 && (
              <path
                className="subflow-edge"
                d={`M ${itemX - gap + 3} ${y + boxH / 2} L ${itemX - 7} ${y + boxH / 2}`}
                markerEnd="url(#arrow)"
              />
            )}
            <rect x={itemX} y={y} width={boxW} height={boxH} rx="8" />
            <foreignObject x={itemX + 10} y={y + 10} width={boxW - 20} height={boxH - 20}>
              <div className="subflow-copy">
                <strong>{item.name}</strong>
                <span>{item.shape}</span>
              </div>
            </foreignObject>
          </g>
        );
      })}
    </g>
  );
}

function TokenStrip({
  label,
  tokens,
  tone,
  visibleCount,
}: {
  label: string;
  tokens: string[];
  tone: 'source' | 'target';
  visibleCount?: number;
}) {
  return (
    <div className="token-strip">
      <span className="strip-label">{label}</span>
      <div className="tokens">
        {tokens.map((token, index) => {
          const pending = visibleCount !== undefined && index >= visibleCount;
          return (
          <span className={`token ${tone} ${pending ? 'pending' : ''}`} key={`${token}-${index}`}>
            {pending ? '...' : token}
          </span>
          );
        })}
      </div>
    </div>
  );
}

function DetailSection({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return (
    <section className="detail-section">
      <div className="section-title">
        {icon}
        <h3>{title}</h3>
      </div>
      <p>{text}</p>
    </section>
  );
}

function ComponentBreakdown({ layerTitle, items }: { layerTitle: string; items: Subcomponent[] }) {
  return (
    <section className="component-breakdown" aria-label="Subcapas relevantes">
      <div className="section-title">
        <Layers size={16} />
        <h3>Subcapas dentro de {layerTitle}</h3>
      </div>
      <div className="subcomponent-list">
        {items.map((item) => (
          <article className="subcomponent-card" key={item.name}>
            <div>
              <strong>{item.name}</strong>
              <span>{item.shape}</span>
            </div>
            <p>{item.role}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function AttentionDeepDive({ visual }: { visual: AttentionVisual }) {
  return (
    <section className="attention-panel attention-deep-dive">
      <div className="panel-heading compact">
        <div>
          <p className="label">Microscopio de atención</p>
          <h2>{visual.title}</h2>
        </div>
        <span className="synthetic-pill">visualización didáctica</span>
      </div>

      <div className="qkv-flow">
        <QkvCard label="Query" tone="query" text={visual.querySource} />
        <QkvCard label="Key" tone="key" text={visual.keySource} />
        <QkvCard label="Value" tone="value" text={visual.valueSource} />
        <div className="attention-equation">
          <span>softmax(QKᵀ / √dₖ) · V</span>
        </div>
      </div>

      <AttentionHeatmap visual={visual} />
    </section>
  );
}

function QkvCard({ label, tone, text }: { label: string; tone: 'query' | 'key' | 'value'; text: string }) {
  return (
    <article className={`qkv-card ${tone}`}>
      <strong>{label}</strong>
      <p>{text}</p>
    </article>
  );
}

function AttentionHeatmap({ visual }: { visual: AttentionVisual }) {
  return (
    <div className="heatmap-wrap">
      <div className="heatmap-grid" style={{ gridTemplateColumns: `130px repeat(${visual.columns.length}, minmax(38px, 1fr))` }}>
        <div className="heatmap-corner">
          {visual.rowsLabel}
          <br />
          {visual.columnsLabel}
        </div>
        {visual.columns.map((column, index) => (
          <div className="heatmap-col" key={`${column}-${index}`}>
            {column}
          </div>
        ))}

        {visual.rows.map((row, rowIndex) => (
          <Fragment key={`${row}-${rowIndex}`}>
            <div className="heatmap-row" key={`${row}-label`}>
              {row}
            </div>
            {visual.matrix[rowIndex].map((value, columnIndex) => (
              <div
                className="heat-cell"
                key={`${row}-${visual.columns[columnIndex]}-${columnIndex}`}
                style={{
                  backgroundColor: `rgba(20, 63, 53, ${0.08 + value * 0.84})`,
                  color: value > 0.65 ? '#ffffff' : '#050505',
                }}
                title={`${row} -> ${visual.columns[columnIndex]}: ${value.toFixed(2)}`}
              >
                {value > 0.6 ? value.toFixed(2) : ''}
              </div>
            ))}
          </Fragment>
        ))}
      </div>
      <p className="heatmap-note">{visual.note}</p>
    </div>
  );
}

export default App;
