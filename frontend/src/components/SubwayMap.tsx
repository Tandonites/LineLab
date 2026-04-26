import { useEffect, useRef, useCallback, useState } from 'react'
import L from 'leaflet'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'
import type { DrawnStation, NewStationDraft, Prediction } from '../App'

// Fix Leaflet default icon paths broken by Vite bundling
// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
})

// ── MTA line-group colours ──────────────────────────────────────────────────
const LINE_COLORS: Record<string, string> = {
  blue:    '#0039A6',
  orange:  '#FF6319',
  yellow:  '#FCCC0A',
  red:     '#EE352E',
  green:   '#00933C',
  brown:   '#996633',
  grey:    '#A7A9AC',
  lime:    '#6CBE45',
  purple:  '#B933AD',
  shuttle: '#808183',
  other:   '#A7A9AC',
}

const LINE_GROUP: Record<string, string> = {
  A:'blue',C:'blue',E:'blue',
  B:'orange',D:'orange',F:'orange',M:'orange',
  N:'yellow',Q:'yellow',R:'yellow',W:'yellow',
  '1':'red','2':'red','3':'red',
  '4':'green','5':'green','6':'green',
  J:'brown',Z:'brown',
  L:'grey',G:'lime','7':'purple',S:'shuttle',
}

function stationColor(lines: string[]): string {
  for (const l of lines) {
    const grp = LINE_GROUP[l]
    if (grp && LINE_COLORS[grp]) return LINE_COLORS[grp]
  }
  return LINE_COLORS.other
}

// ── Station data shape from stations.json ────────────────────────────────────
interface StationData {
  station_complex_id: string
  name: string
  lines: string[]
  borough: string
  lat: number
  lon: number
  total_ridership: number
}

// ── Props ────────────────────────────────────────────────────────────────────
interface Props {
  drawnLine: DrawnStation[]
  prediction: Prediction | null
  newStationDraft: NewStationDraft | null
  onStationClick: (station: DrawnStation) => void
  onMapRightClick: (draft: NewStationDraft) => void
  onCommitNewStation: (name: string) => void
  onCancelDraft: () => void
}

const NYC_CENTER: [number, number] = [40.7128, -73.906]
const NYC_ZOOM = 12
const RESULT_HEAT_PANE = 'result-heat-pane'
const RESULT_BUBBLE_PANE = 'result-bubble-pane'

function impactColor(delta: number): string {
  return delta >= 0 ? '#10b981' : '#ef4444'
}

export default function SubwayMap({
  drawnLine,
  prediction,
  newStationDraft,
  onStationClick,
  onMapRightClick,
  onCommitNewStation,
  onCancelDraft,
}: Props) {
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const stationMarkersRef = useRef<Map<string, L.CircleMarker>>(new Map())
  const drawnPolylineRef = useRef<L.Polyline | null>(null)
  const newStationMarkersRef = useRef<L.Marker[]>([])
  const resultHeatRef = useRef<L.Circle[]>([])
  const resultBubblesRef = useRef<L.CircleMarker[]>([])
  const [stationsData, setStationsData] = useState<StationData[]>([])
  const [draftName, setDraftName] = useState('')

  // ── Load station data ──────────────────────────────────────────────────────
  useEffect(() => {
    fetch('/data/stations.json')
      .then(r => r.json())
      .then((data: StationData[]) => setStationsData(data))
      .catch(console.error)
  }, [])

  // ── Init map once ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return

    const map = L.map(mapContainerRef.current, {
      center: NYC_CENTER,
      zoom: NYC_ZOOM,
      zoomControl: false,
    })

    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      {
        attribution: '© OpenStreetMap © CARTO',
        subdomains: 'abcd',
        maxZoom: 19,
      }
    ).addTo(map)

    // Zoom control bottom-right
    L.control.zoom({ position: 'bottomright' }).addTo(map)

    map.createPane(RESULT_HEAT_PANE)
    map.getPane(RESULT_HEAT_PANE)!.style.zIndex = '430'
    map.createPane(RESULT_BUBBLE_PANE)
    map.getPane(RESULT_BUBBLE_PANE)!.style.zIndex = '460'

    // Context menu → right-click to place new station
    map.on('contextmenu', (e: L.LeafletMouseEvent) => {
      const containerPoint = map.latLngToContainerPoint(e.latlng)
      onMapRightClick({
        lat: e.latlng.lat,
        lon: e.latlng.lng,
        x: containerPoint.x,
        y: containerPoint.y,
      })
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Render existing station markers ───────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map || stationsData.length === 0) return
    const markers = stationMarkersRef.current

    const bounds: [number, number][] = []

    stationsData.forEach(st => {
      const color = stationColor(st.lines)
      const isSelected = drawnLine.some(d => d.id === st.station_complex_id)

      const marker = L.circleMarker([st.lat, st.lon], {
        radius: 6,
        fillColor: color,
        color: isSelected ? '#ffffff' : color,
        weight: isSelected ? 3 : 1,
        fillOpacity: 0.9,
        opacity: 1,
        bubblingMouseEvents: false,
      })

      marker.bindTooltip(
        `<div class="font-semibold">${st.name}</div>` +
        `<div class="text-xs text-gray-300">${st.lines.join(' ')}</div>`,
        { direction: 'top', className: 'leaflet-tooltip-dark' }
      )

      marker.on('click', () => {
        onStationClick({
          id: st.station_complex_id,
          name: st.name,
          lat: st.lat,
          lon: st.lon,
          isNew: false,
          lines: st.lines,
        })
      })

      marker.addTo(map)
      markers.set(st.station_complex_id, marker)
      bounds.push([st.lat, st.lon])
    })

    if (bounds.length > 0) {
      map.fitBounds(bounds as L.LatLngBoundsLiteral, { padding: [40, 40] })
    }

    return () => {
      markers.forEach(m => m.remove())
      markers.clear()
    }
  // We intentionally only init markers once on station data load
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stationsData])

  // ── Update marker rings when drawnLine changes ────────────────────────────
  useEffect(() => {
    const selectedIds = new Set(drawnLine.map(s => s.id))
    stationMarkersRef.current.forEach((marker, id) => {
      const selected = selectedIds.has(id)
      marker.setStyle({
        color: selected ? '#ffffff' : marker.options.fillColor as string,
        weight: selected ? 3 : 1,
      })
    })
  }, [drawnLine])

  // ── Draw the polyline live ─────────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (drawnPolylineRef.current) {
      drawnPolylineRef.current.remove()
      drawnPolylineRef.current = null
    }

    if (drawnLine.length >= 2) {
      const latlngs = drawnLine.map(s => [s.lat, s.lon] as [number, number])
      drawnPolylineRef.current = L.polyline(latlngs, {
        color: '#60a5fa',
        weight: 4,
        opacity: 0.85,
        dashArray: '8 4',
      }).addTo(map)
    }

    // New station markers (orange diamonds via divIcon)
    newStationMarkersRef.current.forEach(m => m.remove())
    newStationMarkersRef.current = []

    drawnLine
      .filter(s => s.isNew)
      .forEach(s => {
        const icon = L.divIcon({
          className: '',
          html: `<div style="
            width:14px;height:14px;
            background:#f97316;
            border:2px solid white;
            transform:rotate(45deg);
            border-radius:2px;
          "></div>`,
          iconSize: [14, 14],
          iconAnchor: [7, 7],
        })
        const m = L.marker([s.lat, s.lon], { icon })
          .bindTooltip(s.name, { direction: 'top' })
          .addTo(map)
        newStationMarkersRef.current.push(m)
      })
  }, [drawnLine])

  // ── Render heat + bubbles for results ─────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    resultHeatRef.current.forEach(layer => layer.remove())
    resultHeatRef.current = []
    resultBubblesRef.current.forEach(layer => layer.remove())
    resultBubblesRef.current = []

    if (!prediction) return

    const maxAbs = Math.max(
      ...prediction.affected_stations.map(s => Math.abs(s.ridership_delta)),
      1
    )

    prediction.affected_stations.forEach(affected => {
      const marker = stationMarkersRef.current.get(affected.station_id)
      if (!marker) return

      const pct = Math.abs(affected.ridership_delta) / maxAbs
      const color = impactColor(affected.ridership_delta)
      const sign = affected.ridership_delta >= 0 ? '+' : ''
      const heatRadiusMeters = 500 + pct * 1200
      const bubbleRadius = 7 + pct * 15

      const outerGlow = L.circle(marker.getLatLng(), {
        radius: heatRadiusMeters,
        pane: RESULT_HEAT_PANE,
        stroke: false,
        fillColor: color,
        fillOpacity: 0.08 + pct * 0.08,
      }).addTo(map)

      const innerGlow = L.circle(marker.getLatLng(), {
        radius: heatRadiusMeters * 0.55,
        pane: RESULT_HEAT_PANE,
        stroke: false,
        fillColor: color,
        fillOpacity: 0.12 + pct * 0.12,
      }).addTo(map)

      const coreGlow = L.circle(marker.getLatLng(), {
        radius: heatRadiusMeters * 0.24,
        pane: RESULT_HEAT_PANE,
        stroke: false,
        fillColor: color,
        fillOpacity: 0.18 + pct * 0.18,
      }).addTo(map)

      const bubble = L.circleMarker(marker.getLatLng(), {
        pane: RESULT_BUBBLE_PANE,
        radius: bubbleRadius,
        fillColor: color,
        fillOpacity: 0.42,
        color: '#f8fafc',
        weight: 2.5,
        opacity: 0.95,
      })
        .bindPopup(
          `<div class="text-sm font-semibold">${affected.name}</div>` +
          `<div class="text-xs">Δ ${sign}${affected.ridership_delta.toLocaleString()} riders/day</div>` +
          `<div class="text-xs">(${sign}${affected.ridership_delta_pct.toFixed(1)}%)</div>`
        )
        .addTo(map)

      resultHeatRef.current.push(outerGlow, innerGlow, coreGlow)
      resultBubblesRef.current.push(bubble)
    })
  }, [prediction])

  // ── Dismiss draft on Escape or outside click ───────────────────────────────
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') onCancelDraft()
  }, [onCancelDraft])

  return (
    <div className="relative w-full h-full">
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Reset view button */}
      <button
        className="absolute top-3 right-3 z-[600] bg-[#1e2029] hover:bg-[#2a2d3a] text-white text-xs px-3 py-1.5 rounded-full border border-gray-600 shadow-lg transition-colors"
        onClick={() => mapRef.current?.setView(NYC_CENTER, NYC_ZOOM)}
      >
        Reset View
      </button>

      {prediction && (
        <div className="absolute bottom-6 left-6 z-[600] rounded-2xl border border-white/10 bg-[#11141d]/90 px-4 py-3 shadow-2xl backdrop-blur-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-gray-300">
            Impact Map
          </p>
          <div className="mt-2 flex items-center gap-2 text-xs text-gray-200">
            <span className="inline-block h-3 w-3 rounded-full border border-white/70 bg-emerald-500/80" />
            Gains
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-gray-200">
            <span className="inline-block h-3 w-3 rounded-full border border-white/70 bg-red-500/80" />
            Losses
          </div>
          <p className="mt-2 max-w-40 text-[11px] leading-relaxed text-gray-400">
            Soft glow shows heat intensity. Bright circles mark affected stations.
          </p>
        </div>
      )}

      {/* New station draft mini-card */}
      {newStationDraft && (
        <div
          className="absolute z-[600] bg-[#1e2029] border border-gray-600 rounded-xl shadow-2xl p-3 w-56"
          style={{ left: Math.min(newStationDraft.x + 12, window.innerWidth - 240), top: newStationDraft.y - 8 }}
          onKeyDown={handleKeyDown}
        >
          <p className="text-white text-xs font-semibold mb-2">New Station</p>
          <input
            autoFocus
            type="text"
            placeholder="Station name…"
            value={draftName}
            onChange={e => setDraftName(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Escape') { setDraftName(''); onCancelDraft() }
              if (e.key === 'Enter' && draftName.trim()) {
                onCommitNewStation(draftName.trim())
                setDraftName('')
              }
            }}
            className="w-full bg-[#0f1117] border border-gray-600 rounded-lg px-2 py-1.5 text-white text-xs outline-none focus:border-blue-500 mb-2"
          />
          <div className="flex gap-2">
            <button
              disabled={!draftName.trim()}
              onClick={() => { onCommitNewStation(draftName.trim()); setDraftName('') }}
              className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs rounded-lg py-1.5 transition-colors"
            >
              Add to Line
            </button>
            <button
              onClick={() => { setDraftName(''); onCancelDraft() }}
              className="flex-1 bg-[#2a2d3a] hover:bg-[#353847] text-gray-300 text-xs rounded-lg py-1.5 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
