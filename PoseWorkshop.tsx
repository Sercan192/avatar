import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAppContext } from '../../contexts/AppContext';
import { useMocapStudio } from '../../hooks/useMocapStudio';
import { useMocapRecorder } from '../../hooks/useMocapRecorder';
import { 
    PlaySquare, Square, Download, Activity, Video, ShieldAlert,
    Smile, Eye, MessageSquare, Music, Cpu, Sliders, RefreshCw, Settings, Sparkles,
    RotateCcw, Shield, Trash2, FolderOpen
} from 'lucide-react';
import { styles } from './DeveloperPanelStyles';
import { GlobalControl } from '../ui/StudioControls';
import { getRotationRange, degToPad, padToDeg } from './puppeteer/PuppeteerUtils';

// Bones where we invert the Z Axis (Slider Down = Positive Value)
const INVERTED_Z_BONES = [
    'LeftShoulder',
    'LeftArm',
    'LeftForeArm',
    'LeftHand'
];

// Bone groups to control
const TARGET_BONES = {
    'LEGS': [
        'LeftUpLeg', 'RightUpLeg',
        'LeftLeg', 'RightLeg',
        'LeftFoot', 'RightFoot'
    ],
    'SPINE': [
        'Hips_Position', // Position Controller (Virtual Bone)
        'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head'
    ],
    'ARMS': [
        'LeftArm', 'RightArm',
        'LeftForeArm', 'RightForeArm',
        'LeftHand', 'RightHand'
    ]
};

// MediaPipe Tasks lists
const VISION_TASKS = [
    { id: 'Face Detector', label: 'Yüz Algılayıcı', icon: Eye },
    { id: 'Face Landmarker', label: 'Yüz İşaretleyici', icon: Smile },
    { id: 'Gesture Recognizer', label: 'Hareket Tanıma', icon: Sparkles },
    { id: 'Hand Landmarker', label: 'El İşaretleyici', icon: Cpu },
    { id: 'Holistic Landmarker', label: 'Bütünsel Dönüm Noktası', icon: Cpu },
    { id: 'Image Segmenter', label: 'Görüntü Segmentleyici', icon: Sliders },
    { id: 'Interactive Segmenter', label: 'Etkileşimli Segmentleyici', icon: Sliders },
    { id: 'Object Detector', label: 'Nesne Dedektörü', icon: Eye },
    { id: 'Pose Landmarker', label: 'Poz İşaretleyici', icon: Video },
];

const AUDIO_TASKS = [
    { id: 'Audio Classifier', label: 'Ses Sınıflandırıcı', icon: Music },
];

const TEXT_TASKS = [
    { id: 'Language Detector', label: 'Dil Dedektörü', icon: MessageSquare },
    { id: 'Text Classifier', label: 'Metin Sınıflandırıcı', icon: MessageSquare },
    { id: 'Text Embedder', label: 'Metin Gömücü', icon: MessageSquare },
];

const PoseWorkshop: React.FC = () => {
    const { 
        riggingParamsRef, idleRiggingParamsRef, saveIdleRiggingParams, 
        saveRiggingParams, addToast, isAnimationPaused, toggleAnimationPause 
    } = useAppContext();

    // MediaPipe Task Selection States
    const [selectedTask, setSelectedTask] = useState<string>('Face Landmarker');
    const [loadingTask, setLoadingTask] = useState<string | null>(null);
    const [delegateType, setDelegateType] = useState<'CPU' | 'GPU'>('GPU');

    const mocap = useMocapStudio(riggingParamsRef, selectedTask, delegateType);
    const recorder = useMocapRecorder(30);

    const [, updateState] = useState<{}>();
    const forceUpdate = useCallback(() => updateState({}), []);

    // Tab states for right panel and rigging
    const [rightPanelTab, setRightPanelTab] = useState<'recorder' | 'rigging'>('recorder');
    const [activeRiggingGroup, setActiveRiggingGroup] = useState<'LEGS' | 'SPINE' | 'ARMS'>('LEGS');

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Bone Value Modifier
    const handleValueChange = (bone: string, axis: 'x' | 'y' | 'z', val: number) => {
        // Mocap Offset check
        const mapping = mocap.mappings.find(m => m.bone === bone && m.axis === axis);
        if (mapping?.enabled) {
            const newMappings = mocap.mappings.map(m => 
                m.id === mapping.id ? { ...m, offset: val } : m
            );
            mocap.setMappings(newMappings);
            return;
        }

        if (!riggingParamsRef.current[bone]) {
            riggingParamsRef.current[bone] = { x: 0, y: 0, z: 0 };
        }
        riggingParamsRef.current[bone][axis] = val;
        forceUpdate();
    };

    // Reset Bone / Reset All
    const handleResetBone = (bone: string) => {
        if (riggingParamsRef.current[bone]) {
            riggingParamsRef.current[bone] = { x: 0, y: 0, z: 0 };
        }
        // Also reset offsets in mocap mappings if exists
        const newMappings = mocap.mappings.map(m => 
            m.bone === bone ? { ...m, offset: 0 } : m
        );
        mocap.setMappings(newMappings);
        forceUpdate();
        addToast(`${bone} reset to default.`, "info");
    };

    const handleResetAll = () => {
        Object.keys(riggingParamsRef.current).forEach(bone => {
            riggingParamsRef.current[bone] = { x: 0, y: 0, z: 0 };
        });
        const newMappings = mocap.mappings.map(m => ({ ...m, enabled: false, offset: 0 }));
        mocap.setMappings(newMappings);
        saveRiggingParams();
        forceUpdate();
        addToast("All bones and mocap connections reset.", "info");
    };

    // Export Settings
    const exportSettings = () => {
        const data = {
            version: "1.2",
            timestamp: new Date().toISOString(),
            riggingParams: riggingParamsRef.current,
            idleRiggingParams: idleRiggingParamsRef.current,
            mocap: {
                mappings: mocap.mappings,
                isDistanceCompEnabled: mocap.isDistanceCompEnabled,
                globalDeadzone: mocap.globalDeadzone
            }
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `avatar_rigging_settings_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        addToast("Rigging parameters exported.", "success");
    };

    // Import Settings
    const importSettings = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target?.result as string);
                
                if (data.riggingParams) {
                    riggingParamsRef.current = data.riggingParams;
                }
                if (data.idleRiggingParams && idleRiggingParamsRef) {
                    idleRiggingParamsRef.current = data.idleRiggingParams;
                }
                if (data.mocap) {
                    if (data.mocap.mappings) mocap.setMappings(data.mocap.mappings);
                    if (data.mocap.isDistanceCompEnabled !== undefined) mocap.setIsDistanceCompEnabled(data.mocap.isDistanceCompEnabled);
                    if (data.mocap.globalDeadzone !== undefined) mocap.setGlobalDeadzone(data.mocap.globalDeadzone);
                }

                saveRiggingParams();
                if (saveIdleRiggingParams) saveIdleRiggingParams();

                forceUpdate();
                addToast("Rigging parameters successfully imported.", "success");
            } catch (err) {
                console.error("Import error:", err);
                addToast("Invalid JSON formatting.", "error");
            }
        };
        reader.readAsText(file);
        event.target.value = ''; // Reset input
    };

    const handleImportClick = () => {
        fileInputRef.current?.click();
    };

    const renderSlider = (bone: string, axis: 'x' | 'y' | 'z', color: string) => {
        // Mocap Mapping Check
        const mapping = mocap.mappings.find(m => m.bone === bone && m.axis === axis);

        // If mocap is active, show the offset value instead of the continuous jittering live value
        // So the slider stays still and only updates when adjusted by the user.
        let val = riggingParamsRef.current[bone]?.[axis] || 0;
        if (mapping?.enabled) {
            val = mapping.offset || 0;
        }
        
        // Hips_Position settings are different (Meters vs Degrees)
        const isPosition = bone === 'Hips_Position';
        const min = isPosition ? -1.5 : -160;
        const max = isPosition ? 1.5 : 160;
        const step = isPosition ? 0.01 : 1;
        const unit = isPosition ? 'm' : '°';

        return (
            <div key={`${bone}-${axis}`} style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', pointerEvents: 'all' }}>
                    <span style={{ width: '15px', color: color, fontSize: '0.7rem', fontWeight: 'bold' }}>{axis.toUpperCase()}</span>
                    <div style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
                        <GlobalControl 
                            min={min} max={max} step={step} 
                            value={val}
                            onChange={(v: number) => handleValueChange(bone, axis, v)}
                            color={color}
                            height={44}
                            size={24}
                        />
                    </div>
                    <span style={{ width: '45px', textAlign: 'right', fontSize: '0.7rem', color: '#E0F2FE', fontFamily: 'monospace' }}>
                        {val.toFixed(isPosition ? 2 : 0)}{unit}
                    </span>
                    
                    {/* MOCAP & ISO TOGGLE BUTTONS */}
                    {mapping && (
                        <div style={{ display: 'flex', gap: '4px' }}>
                            <button 
                                onClick={() => {
                                    const newMappings = mocap.mappings.map(m => 
                                        m.id === mapping.id ? { ...m, enabled: !m.enabled } : m
                                    );
                                    mocap.setMappings(newMappings);
                                }}
                                style={{
                                    padding: '2px 6px',
                                    fontSize: '0.6rem',
                                    borderRadius: '4px',
                                    border: '1px solid ' + (mapping.enabled ? '#10B981' : '#4B5563'),
                                    backgroundColor: mapping.enabled ? 'rgba(16, 185, 129, 0.2)' : 'transparent',
                                    color: mapping.enabled ? '#34D399' : '#94A3B8',
                                    cursor: 'pointer'
                                }}
                                title="Connect to Camera"
                            >
                                {mapping.enabled ? 'LIVE' : 'MOCAP'}
                            </button>
                            
                            {mapping.enabled && (
                                <button 
                                    onClick={() => {
                                        const newMappings = mocap.mappings.map(m => 
                                            m.id === mapping.id ? { ...m, isIsolated: !m.isIsolated } : m
                                        );
                                        mocap.setMappings(newMappings);
                                    }}
                                    style={{
                                        padding: '2px 6px',
                                        fontSize: '0.6rem',
                                        borderRadius: '4px',
                                        border: '1px solid ' + (mapping.isIsolated ? '#8B5CF6' : '#4B5563'),
                                        backgroundColor: mapping.isIsolated ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                                        color: mapping.isIsolated ? '#A78BFA' : '#94A3B8',
                                        cursor: 'pointer'
                                    }}
                                    title="Isolate from Parent Bone Movement (Counter-Rotation)"
                                >
                                    ISO
                                </button>
                            )}
                        </div>
                    )}
                </div>

                {/* MOCAP CONTROLS (If enabled) */}
                {mapping?.enabled && (
                    <div style={{ display: 'flex', gap: '8px', paddingLeft: '23px', alignItems: 'center' }}>
                        <label style={{ fontSize: '0.6rem', color: '#94A3B8', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                            Inv: 
                            <input type="checkbox" checked={mapping.invert} onChange={(e) => {
                                const newMappings = mocap.mappings.map(m => 
                                    m.id === mapping.id ? { ...m, invert: e.target.checked } : m
                                );
                                mocap.setMappings(newMappings);
                            }} />
                        </label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <span style={{ fontSize: '0.6rem', color: '#94A3B8' }}>Sens:</span>
                            <input 
                                type="number" 
                                value={mapping.multiplier} 
                                step={isPosition ? 0.1 : 1} 
                                onChange={(e) => {
                                    const newMappings = mocap.mappings.map(m => 
                                        m.id === mapping.id ? { ...m, multiplier: parseFloat(e.target.value) || 0 } : m
                                    );
                                    mocap.setMappings(newMappings);
                                }} 
                                style={{ width: '32px', fontSize: '0.6rem', background: 'rgba(0,0,0,0.3)', border: '1px solid #4B5563', color: 'white', padding: '1px 2px', borderRadius: '3px' }} 
                            />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flex: 1 }}>
                            <span style={{ fontSize: '0.6rem', color: '#94A3B8' }}>Smooth:</span>
                            <input type="range" min="0" max="0.95" step="0.05" value={mapping.smoothing} onChange={(e) => {
                                const newMappings = mocap.mappings.map(m => 
                                    m.id === mapping.id ? { ...m, smoothing: parseFloat(e.target.value) } : m
                                );
                                mocap.setMappings(newMappings);
                            }} style={{ flex: 1, height: '4px', accentColor: '#8B5CF6' }} />
                        </div>
                    </div>
                )}
            </div>
        );
    };

    // MediaPipe Task Selection States
    const [taskInferenceTime, setTaskInferenceTime] = useState<number>(14.90);
    const [systemFps, setSystemFps] = useState<number>(60);
    const [modelMode, setModelMode] = useState<'standard' | 'custom'>('standard');

    // Hyperparameters states
    const [numFaces, setNumFaces] = useState<number>(1);
    const [minDetectionConfidence, setMinDetectionConfidence] = useState<number>(0.5);
    const [minPresenceConfidence, setMinPresenceConfidence] = useState<number>(0.5);
    const [minTrackingConfidence, setMinTrackingConfidence] = useState<number>(0.5);
    
    const [numHands, setNumHands] = useState<number>(1);
    const [numPoses, setNumPoses] = useState<number>(1);
    const [maxResults, setMaxResults] = useState<number>(5);
    const [scoreThreshold, setScoreThreshold] = useState<number>(0.5);
    const [runningMode, setRunningMode] = useState<'webcam' | 'image'>('webcam');

    // Simulate task loading & model downloading pipeline
    const handleTaskClick = (taskId: string) => {
        if (loadingTask || selectedTask === taskId) return;
        setLoadingTask(taskId);
        
        setTimeout(() => {
            setSelectedTask(taskId);
            setLoadingTask(null);
            
            // Simulating realistic inference overhead values for different models
            if (taskId.includes('Holistic')) {
                setTaskInferenceTime(32.45);
            } else if (taskId.includes('Face Landmarker')) {
                setTaskInferenceTime(11.80);
            } else if (taskId.includes('Pose')) {
                setTaskInferenceTime(21.50);
            } else if (taskId.includes('Audio')) {
                setTaskInferenceTime(7.20);
            } else if (taskId.includes('Detector')) {
                setTaskInferenceTime(18.90);
            } else {
                setTaskInferenceTime(parseFloat((10 + Math.random() * 8).toFixed(2)));
            }
        }, 700);
    };

    useEffect(() => {
        const interval = setInterval(() => {
            const jitter = 58 + Math.random() * 2.5;
            setSystemFps(parseFloat(jitter.toFixed(1)));
        }, 1500);
        return () => clearInterval(interval);
    }, []);

    return (
        <div style={{...styles.overlay}}>
            {/* 1. SOL PANEL (TASK SECTOR) */}
            <div style={{
                position: 'fixed',
                left: 'var(--dev-content-left, 32px)',
                top: 'var(--dev-content-top, 32px)',
                bottom: 'var(--dev-content-bottom, 32px)',
                width: 'var(--dev-panel-left-width, 320px)',
                pointerEvents: 'auto',
                display: 'flex', flexDirection: 'column',
                zIndex: 10,
                backgroundColor: 'var(--dev-module-bg, rgba(2, 6, 23, 0.9))',
                borderRight: 'var(--dev-border, 1px solid rgba(139, 92, 246, 0.3))',
                borderRadius: 'var(--dev-radius, 8px)',
                backdropFilter: 'var(--dev-blur, none)',
                overflow: 'hidden'
            }}>
                <div style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <Activity size={20} color="var(--dev-accent, #10B981)" />
                    <span style={{ fontSize: '0.95rem', fontWeight: 'bold', color: 'var(--dev-accent, #10B981)', letterSpacing: '0.05em' }}>MEDIAPIPE TASKS</span>
                </div>

                <div style={{ 
                    flex: 1, 
                    overflowY: 'auto', 
                    padding: '12px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px'
                }}>
                    {/* VİZYON GÖREVLERİ */}
                    <div>
                        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748B', paddingLeft: '8px', marginBottom: '6px', letterSpacing: '0.05em' }}>
                            VİZYON GÖREVLERİ (VISION)
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            {VISION_TASKS.map((task) => {
                                const Icon = task.icon;
                                const isSelected = selectedTask === task.id;
                                const isLoading = loadingTask === task.id;
                                return (
                                    <button
                                        key={task.id}
                                        onClick={() => handleTaskClick(task.id)}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            width: '100%',
                                            padding: '8px 10px',
                                            backgroundColor: isSelected ? 'rgba(139, 92, 246, 0.15)' : 'transparent',
                                            border: isSelected ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid transparent',
                                            borderRadius: '6px',
                                            color: isSelected ? '#F3E8FF' : '#94A3B8',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s ease',
                                            fontSize: '0.8rem',
                                            textAlign: 'left'
                                        }}
                                        onMouseEnter={(e) => {
                                            if (!isSelected) {
                                                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
                                                e.currentTarget.style.color = '#E2E8F0';
                                            }
                                        }}
                                        onMouseLeave={(e) => {
                                            if (!isSelected) {
                                                e.currentTarget.style.backgroundColor = 'transparent';
                                                e.currentTarget.style.color = '#94A3B8';
                                            }
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <Icon size={14} color={isSelected ? 'var(--dev-accent, #10B981)' : '#64748B'} />
                                            <span>{task.label}</span>
                                        </div>
                                        {isLoading ? (
                                            <RefreshCw size={12} style={{ animation: 'spin 1s linear infinite' }} />
                                        ) : isSelected ? (
                                            <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--dev-accent, #10B981)' }}></div>
                                        ) : null}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* SES GÖREVLERİ */}
                    <div>
                        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748B', paddingLeft: '8px', marginBottom: '6px', letterSpacing: '0.05em' }}>
                            SES GÖREVLERİ (AUDIO)
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            {AUDIO_TASKS.map((task) => {
                                const Icon = task.icon;
                                const isSelected = selectedTask === task.id;
                                const isLoading = loadingTask === task.id;
                                return (
                                    <button
                                        key={task.id}
                                        onClick={() => handleTaskClick(task.id)}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            width: '100%',
                                            padding: '8px 10px',
                                            backgroundColor: isSelected ? 'rgba(139, 92, 246, 0.15)' : 'transparent',
                                            border: isSelected ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid transparent',
                                            borderRadius: '6px',
                                            color: isSelected ? '#F3E8FF' : '#94A3B8',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s ease',
                                            fontSize: '0.8rem',
                                            textAlign: 'left'
                                        }}
                                        onMouseEnter={(e) => {
                                            if (!isSelected) {
                                                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
                                                e.currentTarget.style.color = '#E2E8F0';
                                            }
                                        }}
                                        onMouseLeave={(e) => {
                                            if (!isSelected) {
                                                e.currentTarget.style.backgroundColor = 'transparent';
                                                e.currentTarget.style.color = '#94A3B8';
                                            }
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <Icon size={14} color={isSelected ? 'var(--dev-accent, #10B981)' : '#64748B'} />
                                            <span>{task.label}</span>
                                        </div>
                                        {isLoading ? (
                                            <RefreshCw size={12} style={{ animation: 'spin 1s linear infinite' }} />
                                        ) : isSelected ? (
                                            <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--dev-accent, #10B981)' }}></div>
                                        ) : null}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* METİN GÖREVLERİ */}
                    <div>
                        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748B', paddingLeft: '8px', marginBottom: '6px', letterSpacing: '0.05em' }}>
                            METİN GÖREVLERİ (TEXT)
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            {TEXT_TASKS.map((task) => {
                                const Icon = task.icon;
                                const isSelected = selectedTask === task.id;
                                const isLoading = loadingTask === task.id;
                                return (
                                    <button
                                        key={task.id}
                                        onClick={() => handleTaskClick(task.id)}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            width: '100%',
                                            padding: '8px 10px',
                                            backgroundColor: isSelected ? 'rgba(139, 92, 246, 0.15)' : 'transparent',
                                            border: isSelected ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid transparent',
                                            borderRadius: '6px',
                                            color: isSelected ? '#F3E8FF' : '#94A3B8',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s ease',
                                            fontSize: '0.8rem',
                                            textAlign: 'left'
                                        }}
                                        onMouseEnter={(e) => {
                                            if (!isSelected) {
                                                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
                                                e.currentTarget.style.color = '#E2E8F0';
                                            }
                                        }}
                                        onMouseLeave={(e) => {
                                            if (!isSelected) {
                                                e.currentTarget.style.backgroundColor = 'transparent';
                                                e.currentTarget.style.color = '#94A3B8';
                                            }
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <Icon size={14} color={isSelected ? 'var(--dev-accent, #10B981)' : '#64748B'} />
                                            <span>{task.label}</span>
                                        </div>
                                        {isLoading ? (
                                            <RefreshCw size={12} style={{ animation: 'spin 1s linear infinite' }} />
                                        ) : isSelected ? (
                                            <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--dev-accent, #10B981)' }}></div>
                                        ) : null}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>

            {/* 2. ORTA PANEL (SETTINGS SECTOR - PARAMETRELER & TELEMETRİ) */}
            <div style={{
                position: 'fixed',
                left: 'calc(var(--dev-content-left, 32px) + var(--dev-panel-left-width, 320px) + 12px)',
                top: 'var(--dev-content-top, 32px)',
                bottom: 'var(--dev-content-bottom, 32px)',
                width: '320px',
                pointerEvents: 'auto',
                display: 'flex', flexDirection: 'column',
                zIndex: 10,
                backgroundColor: 'var(--dev-module-bg, rgba(2, 6, 23, 0.9))',
                borderRight: 'var(--dev-border, 1px solid rgba(139, 92, 246, 0.3))',
                borderRadius: 'var(--dev-radius, 8px)',
                backdropFilter: 'var(--dev-blur, none)',
                overflow: 'hidden'
            }}>
                <div style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <Settings size={18} color="var(--dev-accent, #10B981)" />
                    <span style={{ fontSize: '0.95rem', fontWeight: 'bold', color: 'var(--dev-accent, #10B981)', letterSpacing: '0.05em' }}>MODEL & PARAMETRELER</span>
                </div>

                <div style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px'
                }}>
                    {/* MODEL SELECTION */}
                    <div>
                        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748B', marginBottom: '8px', letterSpacing: '0.05em' }}>
                            MODEL SEÇİMİ (MODEL SELECTION)
                        </div>
                        <div style={{ display: 'flex', gap: '6px', marginBottom: '10px' }}>
                            <button
                                onClick={() => setModelMode('standard')}
                                style={{
                                    flex: 1,
                                    padding: '6px 12px',
                                    fontSize: '0.75rem',
                                    backgroundColor: modelMode === 'standard' ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                                    border: modelMode === 'standard' ? '1px solid rgba(139, 92, 246, 0.5)' : '1px solid rgba(255,255,255,0.05)',
                                    borderRadius: '6px',
                                    color: modelMode === 'standard' ? '#F3E8FF' : '#94A3B8',
                                    cursor: 'pointer'
                                }}
                            >
                                Standart
                            </button>
                            <button
                                onClick={() => setModelMode('custom')}
                                style={{
                                    flex: 1,
                                    padding: '6px 12px',
                                    fontSize: '0.75rem',
                                    backgroundColor: modelMode === 'custom' ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                                    border: modelMode === 'custom' ? '1px solid rgba(139, 92, 246, 0.5)' : '1px solid rgba(255,255,255,0.05)',
                                    borderRadius: '6px',
                                    color: modelMode === 'custom' ? '#F3E8FF' : '#94A3B8',
                                    cursor: 'pointer'
                                }}
                            >
                                Özel Model (Upload)
                            </button>
                        </div>
                        <div style={{
                            backgroundColor: 'rgba(0,0,0,0.2)',
                            padding: '8px 10px',
                            borderRadius: '6px',
                            fontSize: '0.7rem',
                            color: '#94A3B8',
                            border: '1px solid rgba(255,255,255,0.02)',
                            lineHeight: '1.4'
                        }}>
                            {modelMode === 'standard' ? (
                                <div>
                                    <span style={{ color: '#F3E8FF', fontWeight: '500' }}>Aktif Model:</span> {selectedTask.toLowerCase().replace(/ /g, '_')}_v2.task
                                </div>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                    <span>Yüklemek istediğiniz .task veya .tflite dosyasını sürükleyin:</span>
                                    <button style={{
                                        padding: '4px 8px',
                                        backgroundColor: 'rgba(255,255,255,0.05)',
                                        border: '1px dashed rgba(255,255,255,0.15)',
                                        borderRadius: '4px',
                                        color: '#E2E8F0',
                                        cursor: 'pointer',
                                        fontSize: '0.65rem'
                                    }}>
                                        Dosya Seç...
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* HYPERPARAMETERS SECTOR */}
                    <div>
                        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748B', marginBottom: '8px', letterSpacing: '0.05em' }}>
                            HİPERPARAMETRELER (HYPERPARAMETERS)
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                            {/* Face landmarker fields */}
                            {(selectedTask === 'Face Landmarker' || selectedTask === 'Face Detector') && (
                                <>
                                    {selectedTask === 'Face Landmarker' && (
                                        <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Num Faces (Yüz Sayısı)</span>
                                                <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{numFaces}</span>
                                            </div>
                                            <input
                                                type="range" min="1" max="4" step="1"
                                                value={numFaces} onChange={(e) => setNumFaces(parseInt(e.target.value))}
                                                style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                            />
                                        </div>
                                    )}

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Min Detection Confidence</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minDetectionConfidence.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={minDetectionConfidence} onChange={(e) => setMinDetectionConfidence(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>

                                    {selectedTask === 'Face Landmarker' && (
                                        <>
                                            <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                                    <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Min Presence Confidence</span>
                                                    <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minPresenceConfidence.toFixed(2)}</span>
                                                </div>
                                                <input
                                                    type="range" min="0.1" max="1.0" step="0.05"
                                                    value={minPresenceConfidence} onChange={(e) => setMinPresenceConfidence(parseFloat(e.target.value))}
                                                    style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                                />
                                            </div>

                                            <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                                    <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Min Tracking Confidence</span>
                                                    <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minTrackingConfidence.toFixed(2)}</span>
                                                </div>
                                                <input
                                                    type="range" min="0.1" max="1.0" step="0.05"
                                                    value={minTrackingConfidence} onChange={(e) => setMinTrackingConfidence(parseFloat(e.target.value))}
                                                    style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                                />
                                            </div>
                                        </>
                                    )}
                                </>
                            )}

                            {/* Gesture recognizer and hand landmarker */}
                            {(selectedTask === 'Gesture Recognizer' || selectedTask === 'Hand Landmarker') && (
                                <>
                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Num Hands (El Sayısı)</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{numHands}</span>
                                        </div>
                                        <input
                                            type="range" min="1" max="4" step="1"
                                            value={numHands} onChange={(e) => setNumHands(parseInt(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Min Hand Detection Conf.</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minDetectionConfidence.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={minDetectionConfidence} onChange={(e) => setMinDetectionConfidence(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Min Tracking Confidence</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minTrackingConfidence.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={minTrackingConfidence} onChange={(e) => setMinTrackingConfidence(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>
                                </>
                            )}

                            {/* Pose landmarker fields */}
                            {selectedTask === 'Pose Landmarker' && (
                                <>
                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Num Poses (Poz Sayısı)</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{numPoses}</span>
                                        </div>
                                        <input
                                            type="range" min="1" max="4" step="1"
                                            value={numPoses} onChange={(e) => setNumPoses(parseInt(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Min Detection Confidence</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minDetectionConfidence.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={minDetectionConfidence} onChange={(e) => setMinDetectionConfidence(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Min Presence Confidence</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minPresenceConfidence.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={minPresenceConfidence} onChange={(e) => setMinPresenceConfidence(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>
                                </>
                            )}

                            {/* Holistic Landmarker */}
                            {selectedTask === 'Holistic Landmarker' && (
                                <>
                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Face Detection Confidence</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minDetectionConfidence.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={minDetectionConfidence} onChange={(e) => setMinDetectionConfidence(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Pose Detection Confidence</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{minPresenceConfidence.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={minPresenceConfidence} onChange={(e) => setMinPresenceConfidence(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>
                                </>
                            )}

                            {/* Audio, Object detector or Segmenters */}
                            {(selectedTask.includes('Audio') || selectedTask.includes('Detector') || selectedTask.includes('Segmenter') || selectedTask.includes('Text') || selectedTask.includes('Language')) && (
                                <>
                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Max Results (Maks Sonuç)</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{maxResults}</span>
                                        </div>
                                        <input
                                            type="range" min="1" max="10" step="1"
                                            value={maxResults} onChange={(e) => setMaxResults(parseInt(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Score Threshold (Eşik)</span>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{scoreThreshold.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range" min="0.1" max="1.0" step="0.05"
                                            value={scoreThreshold} onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
                                            style={{ width: '100%', accentColor: 'var(--dev-accent, #10B981)' }}
                                        />
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    {/* HARDWARE & RUNNING MODES */}
                    <div>
                        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748B', marginBottom: '8px', letterSpacing: '0.05em' }}>
                            DONANIM & MOD AYARLARI
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {/* Delegate selection */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Donanım Delegate</span>
                                <div style={{ display: 'flex', gap: '4px', backgroundColor: 'rgba(0,0,0,0.2)', padding: '2px', borderRadius: '4px' }}>
                                    <button
                                        onClick={() => setDelegateType('CPU')}
                                        style={{
                                            padding: '3px 8px',
                                            fontSize: '0.65rem',
                                            backgroundColor: delegateType === 'CPU' ? 'rgba(255,255,255,0.08)' : 'transparent',
                                            border: 'none',
                                            borderRadius: '3px',
                                            color: delegateType === 'CPU' ? '#FFF' : '#64748B',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        CPU
                                    </button>
                                    <button
                                        onClick={() => setDelegateType('GPU')}
                                        style={{
                                            padding: '3px 8px',
                                            fontSize: '0.65rem',
                                            backgroundColor: delegateType === 'GPU' ? 'var(--dev-accent, #10B981)' : 'transparent',
                                            border: 'none',
                                            borderRadius: '3px',
                                            color: delegateType === 'GPU' ? '#020617' : '#64748B',
                                            cursor: 'pointer',
                                            fontWeight: delegateType === 'GPU' ? 'bold' : 'normal'
                                        }}
                                    >
                                        GPU
                                    </button>
                                </div>
                            </div>

                            {/* Running Mode selection */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Çalışma Modu</span>
                                <div style={{ display: 'flex', gap: '4px', backgroundColor: 'rgba(0,0,0,0.2)', padding: '2px', borderRadius: '4px' }}>
                                    <button
                                        onClick={() => setRunningMode('webcam')}
                                        style={{
                                            padding: '3px 8px',
                                            fontSize: '0.65rem',
                                            backgroundColor: runningMode === 'webcam' ? 'rgba(255,255,255,0.08)' : 'transparent',
                                            border: 'none',
                                            borderRadius: '3px',
                                            color: runningMode === 'webcam' ? '#FFF' : '#64748B',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        Webcam
                                    </button>
                                    <button
                                        onClick={() => setRunningMode('image')}
                                        style={{
                                            padding: '3px 8px',
                                            fontSize: '0.65rem',
                                            backgroundColor: runningMode === 'image' ? 'rgba(255,255,255,0.08)' : 'transparent',
                                            border: 'none',
                                            borderRadius: '3px',
                                            color: runningMode === 'image' ? '#FFF' : '#64748B',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        Image
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* TELEMETRY & PERFORMANCE MONITOR */}
                    <div style={{ marginTop: 'auto', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748B', marginBottom: '8px', letterSpacing: '0.05em' }}>
                            TELEMETRİ (PERFORMANCE MONITOR)
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                                <span style={{ color: '#94A3B8' }}>Inference Time:</span>
                                <span style={{ fontFamily: 'monospace', color: 'var(--dev-accent, #10B981)', fontWeight: 'bold' }}>{taskInferenceTime.toFixed(2)} ms</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                                <span style={{ color: '#94A3B8' }}>Queue FPS:</span>
                                <span style={{ fontFamily: 'monospace', color: '#38BDF8', fontWeight: 'bold' }}>{systemFps} FPS</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                                <span style={{ color: '#94A3B8' }}>Hardware Accel:</span>
                                <span style={{ color: delegateType === 'GPU' ? '#34D399' : '#94A3B8', fontSize: '0.7rem' }}>
                                    {delegateType === 'GPU' ? 'Active: WebGL2' : 'WASM Multithread'}
                                </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                                <span style={{ color: '#94A3B8' }}>VRAM Buffer:</span>
                                <span style={{ fontFamily: 'monospace', color: '#F472B6' }}>256MB allocated</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* 3. SAĞ PANEL (Mocap Studio & Recorder) */}
            <div style={{ 
                position: 'fixed',
                right: 'var(--dev-content-right, 32px)',
                top: 'var(--dev-content-top, 32px)',
                bottom: 'var(--dev-content-bottom, 32px)',
                width: 'var(--dev-panel-right-width, 320px)',
                pointerEvents: 'auto', 
                display: 'flex', 
                flexDirection: 'column', 
                zIndex: 10, 
                backgroundColor: 'var(--dev-module-bg, rgba(2, 6, 23, 0.9))',
                borderLeft: 'var(--dev-border, 1px solid rgba(139, 92, 246, 0.3))',
                borderRadius: 'var(--dev-radius, 8px)',
                backdropFilter: 'var(--dev-blur, none)',
                overflow: 'hidden'
            }}>
                {/* Header */}
                <div style={{ padding: '16px 16px 12px 16px', display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0 }}>
                    <Video size={18} color="var(--dev-accent, #A78BFA)" />
                    <span style={{ fontSize: '1rem', fontWeight: 'bold', color: 'var(--dev-accent, #A78BFA)' }}>Studio & Controls</span>
                </div>

                {/* Sub Panel Tab Selector */}
                <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.05)', padding: '0 8px', flexShrink: 0, gap: '4px' }}>
                    <button 
                        onClick={() => setRightPanelTab('recorder')}
                        style={{
                            flex: 1,
                            padding: '10px 4px',
                            fontSize: '0.75rem',
                            fontWeight: rightPanelTab === 'recorder' ? 'bold' : 'normal',
                            color: rightPanelTab === 'recorder' ? 'var(--dev-accent, #10B981)' : '#94A3B8',
                            border: 'none',
                            borderBottom: rightPanelTab === 'recorder' ? '2px solid var(--dev-accent, #10B981)' : '2px solid transparent',
                            backgroundColor: 'transparent',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                        }}
                    >
                        Recorder & Cam
                    </button>
                    <button 
                        onClick={() => setRightPanelTab('rigging')}
                        style={{
                            flex: 1,
                            padding: '10px 4px',
                            fontSize: '0.75rem',
                            fontWeight: rightPanelTab === 'rigging' ? 'bold' : 'normal',
                            color: rightPanelTab === 'rigging' ? 'var(--dev-accent, #10B981)' : '#94A3B8',
                            border: 'none',
                            borderBottom: rightPanelTab === 'rigging' ? '2px solid var(--dev-accent, #10B981)' : '2px solid transparent',
                            backgroundColor: 'transparent',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                        }}
                    >
                        Bone Rigging
                    </button>
                </div>

                <div style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '12px',
                    color: '#E0F2FE',
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '12px'
                }}>
                    {rightPanelTab === 'recorder' ? (
                        <>
                            {/* MOCAP VIDEO & CORE CONTROLS */}
                            <div style={{
                                backgroundColor: 'rgba(255,255,255,0.02)',
                                borderRadius: 'var(--dev-radius, 8px)',
                                padding: '12px',
                                border: '1px solid rgba(255,255,255,0.05)'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                    <h4 style={{ margin: 0, fontSize: '0.85rem', color: '#E0F2FE', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        Sensor Feed {mocap.isActive && <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--dev-accent, #10B981)', animation: 'pulse 2s infinite' }}></span>}
                                    </h4>
                                    
                                    <div style={{ display: 'flex', gap: '6px' }}>
                                        {!mocap.isActive ? (
                                            <button 
                                                onClick={mocap.startMocap}
                                                style={{...styles.actionButton, backgroundColor: 'rgba(59, 130, 246, 0.8)', padding: '4px 10px', fontSize: '0.75rem'}}
                                            >
                                                Start
                                            </button>
                                        ) : (
                                            <>
                                                <button 
                                                    onClick={mocap.calibrate}
                                                    style={{...styles.actionButton, backgroundColor: 'rgba(245, 158, 11, 0.2)', color: '#F59E0B', border: '1px solid rgba(245, 158, 11, 0.5)', padding: '4px 10px', fontSize: '0.75rem'}}
                                                >
                                                    {mocap.countdown !== null ? `Calib (${mocap.countdown})` : 'T-Pose'}
                                                </button>
                                                <button 
                                                    onClick={mocap.stopMocap}
                                                    style={{...styles.actionButton, backgroundColor: 'rgba(239, 68, 68, 0.2)', color: '#EF4444', border: '1px solid rgba(239, 68, 68, 0.5)', padding: '4px 10px', fontSize: '0.75rem'}}
                                                >
                                                    Stop
                                                </button>
                                            </>
                                        )}
                                    </div>
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    <div style={{
                                        position: 'relative',
                                        width: '100%',
                                        aspectRatio: '4/3',
                                        backgroundColor: '#000',
                                        borderRadius: '6px',
                                        overflow: 'hidden',
                                        border: '1px solid rgba(255,255,255,0.05)'
                                    }}>
                                        <video 
                                            ref={mocap.videoRef} 
                                            style={{ 
                                                width: '100%', height: '100%', objectFit: 'cover',
                                                transform: 'scaleX(-1)' // Ayna efekti
                                            }} 
                                            width={640}
                                            height={480}
                                            playsInline 
                                            autoPlay 
                                            muted 
                                        />
                                        <canvas 
                                            ref={mocap.canvasRef}
                                            style={{
                                                position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                                                objectFit: 'cover',
                                                transform: 'scaleX(-1)'
                                            }}
                                        />
                                        {mocap.error && (
                                            <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(239, 68, 68, 0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', padding: '16px', textAlign: 'center', fontSize: '0.75rem' }}>
                                                <ShieldAlert size={20} style={{ marginBottom: '8px' }} />
                                                <br/>{mocap.error}
                                            </div>
                                        )}
                                    </div>
                                    
                                    <div style={{ display: 'flex', gap: '8px' }}>
                                        <div style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                            <div style={{ fontSize: '0.65rem', color: '#9CA3AF', marginBottom: '2px' }}>Status</div>
                                            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: mocap.isReady ? 'var(--dev-accent, #10B981)' : '#F59E0B' }}>
                                                {mocap.isReady ? 'Ready' : 'Pending'}
                                            </div>
                                        </div>
                                        <div style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                            <div style={{ fontSize: '0.65rem', color: '#9CA3AF', marginBottom: '2px' }}>Links</div>
                                            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--dev-accent, #3B82F6)' }}>
                                                {mocap.mappings.filter(m => m.enabled).length} Tracked
                                            </div>
                                        </div>
                                    </div>

                                    {/* TASK INTELLIGENCE INTERACTIVE PANEL */}
                                    <div style={{
                                        backgroundColor: 'rgba(15, 23, 42, 0.4)',
                                        border: '1px solid rgba(139, 92, 246, 0.2)',
                                        borderRadius: '6px',
                                        padding: '10px',
                                        marginTop: '4px'
                                    }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                            <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#A78BFA', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                                Task Intelligence Output
                                            </div>
                                            {mocap.taskOutput && (
                                                <div style={{
                                                    fontSize: '0.55rem',
                                                    padding: '1px 5px',
                                                    borderRadius: '4px',
                                                    fontWeight: 'bold',
                                                    backgroundColor: mocap.taskOutput.isRealModel ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                                                    color: mocap.taskOutput.isRealModel ? '#10B981' : '#F59E0B',
                                                    border: mocap.taskOutput.isRealModel ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(245, 158, 11, 0.3)',
                                                    textTransform: 'uppercase'
                                                }}>
                                                    {mocap.taskOutput.isRealModel ? 'Live AI Model' : 'Simulated Backup'}
                                                </div>
                                            )}
                                        </div>

                                        {/* TEXT TASKS INTERACTIVE FIELD */}
                                        {(selectedTask.includes('Text') || selectedTask.includes('Language') || selectedTask.includes('Metin') || selectedTask.includes('Dil')) && (
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '8px' }}>
                                                <div style={{ fontSize: '0.65rem', color: '#94A3B8' }}>Giriş Metni (Real-time Evaluation):</div>
                                                <div style={{ display: 'flex', gap: '4px' }}>
                                                    <input
                                                        type="text"
                                                        value={mocap.textInput}
                                                        onChange={(e) => {
                                                            mocap.setTextInput(e.target.value);
                                                            mocap.processTextInput(selectedTask, e.target.value);
                                                        }}
                                                        placeholder="Merhaba veya I love this avatar! yazın..."
                                                        style={{
                                                            flex: 1,
                                                            fontSize: '0.75rem',
                                                            backgroundColor: 'rgba(0,0,0,0.3)',
                                                            border: '1px solid #4B5563',
                                                            borderRadius: '4px',
                                                            padding: '4px 8px',
                                                            color: 'white'
                                                        }}
                                                    />
                                                </div>
                                            </div>
                                        )}

                                        {mocap.taskOutput ? (
                                            <div style={{ fontFamily: 'monospace', fontSize: '0.7rem', color: '#E2E8F0', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                                {mocap.taskOutput.type === 'sentiment' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Duygu Analizi:</span>
                                                            <span style={{ color: '#F43F5E', fontWeight: 'bold' }}>{mocap.taskOutput.emotion}</span>
                                                        </div>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Güven Skoru:</span>
                                                            <span style={{ color: '#10B981' }}>%{(mocap.taskOutput.score * 100).toFixed(0)}</span>
                                                        </div>
                                                        <div style={{ fontSize: '0.65rem', color: '#6B7280', marginTop: '4px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '4px' }}>
                                                            * Duygu durumuna göre avatar ifadeleri eş zamanlı tetiklenir.
                                                        </div>
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'language' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                            <span style={{ color: '#94A3B8' }}>Algılanan Dil:</span>
                                                            <span style={{ color: '#38BDF8', fontWeight: 'bold' }}>{mocap.taskOutput.flag} {mocap.taskOutput.detected}</span>
                                                        </div>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Doğruluk Oranı:</span>
                                                            <span style={{ color: '#10B981' }}>%{(mocap.taskOutput.confidence * 100).toFixed(1)}</span>
                                                        </div>
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'embedding' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Embedding Vektörü:</span>
                                                            <span style={{ color: '#F43F5E' }}>[{mocap.taskOutput.coords.x.toFixed(2)}, {mocap.taskOutput.coords.y.toFixed(2)}, {mocap.taskOutput.coords.z.toFixed(2)}]</span>
                                                        </div>
                                                        <div style={{ fontSize: '0.65rem', color: '#6B7280', marginTop: '4px' }}>
                                                            Komşu boyutlar hesaplandı. Constellation haritası güncellendi.
                                                        </div>
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'audio' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Frekans Sınıfı:</span>
                                                            <span style={{ color: '#34D399', fontWeight: 'bold' }}>{mocap.taskOutput.classification}</span>
                                                        </div>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Ses Yoğunluğu:</span>
                                                            <span style={{ color: '#38BDF8' }}>{Math.round(mocap.taskOutput.volume * 100)} dB</span>
                                                        </div>
                                                        <div style={{ fontSize: '0.65rem', color: '#6B7280', marginTop: '4px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '4px' }}>
                                                            * Ses şiddeti avatarın çene açısını (jawOpen) doğrudan kontrol eder.
                                                        </div>
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'face' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Yüz Noktası:</span>
                                                            <span style={{ color: '#10B981' }}>{mocap.taskOutput.landmarksCount} Marks</span>
                                                        </div>
                                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', marginTop: '4px' }}>
                                                            <span style={{ color: '#64748B', fontSize: '0.65rem' }}>Active Blendshapes:</span>
                                                            {mocap.taskOutput.blendshapes && mocap.taskOutput.blendshapes.slice(0, 3).map((b: any, idx: number) => (
                                                                <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', paddingLeft: '4px' }}>
                                                                    <span>{b.categoryName}:</span>
                                                                    <span style={{ color: '#A78BFA' }}>{b.score.toFixed(3)}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'gesture' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Hareket Sınıfı:</span>
                                                            <span style={{ color: '#F97316', fontWeight: 'bold' }}>{mocap.taskOutput.gestureName.toUpperCase()}</span>
                                                        </div>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Güvenirlik:</span>
                                                            <span style={{ color: '#10B981' }}>%{(mocap.taskOutput.score * 100).toFixed(0)}</span>
                                                        </div>
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'object' && (
                                                    <>
                                                        <div style={{ color: '#94A3B8', marginBottom: '2px' }}>Tespit Edilen Nesneler:</div>
                                                        {mocap.taskOutput.detections && mocap.taskOutput.detections.map((d: any, idx: number) => (
                                                            <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', paddingLeft: '4px' }}>
                                                                <span>{d.categories[0].categoryName}:</span>
                                                                <span style={{ color: '#38BDF8' }}>%{(d.categories[0].score * 100).toFixed(0)}</span>
                                                            </div>
                                                        ))}
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'faceDetector' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Tespit Edilen Yüzler:</span>
                                                            <span style={{ color: '#F43F5E', fontWeight: 'bold' }}>{mocap.taskOutput.detectionsCount}</span>
                                                        </div>
                                                    </>
                                                )}
                                                {mocap.taskOutput.type === 'hand' && (
                                                    <>
                                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span style={{ color: '#94A3B8' }}>Takip Edilen Eller:</span>
                                                            <span style={{ color: '#F97316', fontWeight: 'bold' }}>{mocap.taskOutput.activeHandsCount}</span>
                                                        </div>
                                                    </>
                                                )}
                                            </div>
                                        ) : (
                                            <div style={{ fontSize: '0.65rem', color: '#64748B', fontStyle: 'italic', textAlign: 'center', padding: '6px 0' }}>
                                                Alıcı verisi işleniyor... (Inference stream active)
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* FİLTRELEME VE HASSASİYET AYARLARI */}
                            <div style={{
                                backgroundColor: 'rgba(255,255,255,0.02)',
                                borderRadius: 'var(--dev-radius, 8px)',
                                padding: '12px',
                                border: '1px solid rgba(255,255,255,0.05)'
                            }}>
                                <h4 style={{ margin: 0, marginBottom: '12px', fontSize: '0.85rem', color: '#E0F2FE', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <Activity size={14} /> 
                                    Filtering & Params
                                </h4>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                            <label style={{ fontSize: '0.75rem', color: '#9CA3AF' }}>Global Deadzone</label>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--dev-accent, #38BDF8)' }}>{mocap.globalDeadzone.toFixed(3)}</span>
                                        </div>
                                        <input 
                                            type="range" 
                                            min="0" max="0.1" step="0.001" 
                                            value={mocap.globalDeadzone} 
                                            onChange={(e) => mocap.setGlobalDeadzone(parseFloat(e.target.value))}
                                            style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--dev-accent)' }}
                                        />
                                        <p style={{ fontSize: '0.6rem', color: '#6B7280', marginTop: '6px', margin: 0 }}>
                                            Filters out micro-jitters from the camera sensor.
                                        </p>
                                    </div>

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                                            <input 
                                                type="checkbox" 
                                                checked={mocap.isDistanceCompEnabled}
                                                onChange={(e) => mocap.setIsDistanceCompEnabled(e.target.checked)}
                                                style={{ width: '14px', height: '14px', accentColor: 'var(--dev-accent)' }}
                                            />
                                            <span style={{ fontSize: '0.75rem', color: '#E0F2FE' }}>Depth Compensation</span>
                                        </label>
                                        <p style={{ fontSize: '0.6rem', color: '#6B7280', marginTop: '6px', margin: 0 }}>
                                            Dynamically scales arm movements based on shoulder width.
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* TRUE ANIMATION RECORDER */}
                            <div style={{
                                backgroundColor: 'rgba(255,255,255,0.02)',
                                borderRadius: 'var(--dev-radius, 8px)',
                                padding: '12px',
                                border: recorder.isRecording ? '1px solid #EF4444' : '1px solid rgba(255,255,255,0.05)'
                            }}>
                                <h4 style={{ margin: 0, marginBottom: '12px', fontSize: '0.85rem', color: '#E0F2FE', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    🎬 Native Recorder
                                    {recorder.isRecording && <span style={{ fontSize: '0.7rem', color: '#EF4444', animation: 'pulse 1.5s infinite' }}>● REC</span>}
                                </h4>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    {!recorder.isRecording ? (
                                        <button 
                                            onClick={recorder.startRecording}
                                            disabled={!mocap.isReady}
                                            style={{
                                                ...styles.actionButton,
                                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                                                backgroundColor: mocap.isReady ? '#EF4444' : 'rgba(255,255,255,0.1)', 
                                                color: mocap.isReady ? '#fff' : '#6B7280', 
                                                padding: '10px 16px', fontSize: '0.8rem', fontWeight: 600, 
                                                cursor: mocap.isReady ? 'pointer' : 'not-allowed', width: '100%'
                                            }}
                                        >
                                            <PlaySquare size={16} />
                                            Start Record
                                        </button>
                                    ) : (
                                        <button 
                                            onClick={recorder.stopRecording}
                                            style={{
                                                ...styles.actionButton,
                                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                                                backgroundColor: 'rgba(0,0,0,0.5)', 
                                                color: '#EF4444', border: '1px solid #EF4444', 
                                                padding: '10px 16px', fontSize: '0.8rem', fontWeight: 600, 
                                                cursor: 'pointer', width: '100%'
                                            }}
                                        >
                                            <Square size={16} fill="#EF4444" />
                                            Stop Record
                                        </button>
                                    )}

                                    <div style={{ backgroundColor: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <div style={{ fontSize: '0.7rem', color: '#9CA3AF' }}>Captured Frames</div>
                                            <div style={{ fontSize: '1rem', fontWeight: 'bold', color: 'var(--dev-accent, #E0F2FE)', fontFamily: 'monospace' }}>
                                                {recorder.frameCount}
                                            </div>
                                        </div>
                                        
                                        <button 
                                            onClick={recorder.exportToNativeJson}
                                            disabled={recorder.frameCount === 0 || recorder.isRecording}
                                            style={{
                                                ...styles.actionButton,
                                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                                                backgroundColor: (recorder.frameCount > 0 && !recorder.isRecording) ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255,255,255,0.05)',
                                                color: (recorder.frameCount > 0 && !recorder.isRecording) ? '#34D399' : '#6B7280',
                                                border: (recorder.frameCount > 0 && !recorder.isRecording) ? '1px solid rgba(16, 185, 129, 0.5)' : '1px solid transparent',
                                                padding: '6px 12px', fontSize: '0.75rem', fontWeight: 600, 
                                                cursor: (recorder.frameCount > 0 && !recorder.isRecording) ? 'pointer' : 'not-allowed', width: '100%'
                                            }}
                                        >
                                            <Download size={14} />
                                            Export JSON
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </>
                    ) : (
                        <>
                            {/* BONE RIGGING INTERFACE */}
                            {/* Sub Tab selector for Bone groups */}
                            <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
                                {(Object.keys(TARGET_BONES) as Array<keyof typeof TARGET_BONES>).map(group => (
                                    <button
                                        key={group}
                                        onClick={() => setActiveRiggingGroup(group)}
                                        style={{
                                            flex: 1,
                                            padding: '6px 2px',
                                            fontSize: '0.65rem',
                                            fontWeight: activeRiggingGroup === group ? 'bold' : 'normal',
                                            backgroundColor: activeRiggingGroup === group ? 'rgba(139, 92, 246, 0.25)' : 'rgba(255,255,255,0.02)',
                                            border: activeRiggingGroup === group ? '1px solid rgba(139, 92, 246, 0.5)' : '1px solid rgba(255,255,255,0.05)',
                                            borderRadius: '6px',
                                            color: activeRiggingGroup === group ? '#F3E8FF' : '#94A3B8',
                                            cursor: 'pointer',
                                            textAlign: 'center',
                                            transition: 'all 0.15s'
                                        }}
                                    >
                                        {group}
                                    </button>
                                ))}
                            </div>

                            {/* Freeze / Unfreeze Avatar Control */}
                            <div style={{ display: 'flex', gap: '4px', margin: '4px 0' }}>
                                <button 
                                    onClick={toggleAnimationPause}
                                    style={{
                                        ...styles.actionButton,
                                        flex: 1,
                                        backgroundColor: isAnimationPaused ? 'rgba(245, 158, 11, 0.2)' : 'rgba(255,255,255,0.03)',
                                        border: isAnimationPaused ? '1px solid #F59E0B' : '1px solid rgba(255,255,255,0.05)',
                                        color: isAnimationPaused ? '#FBBF24' : '#fff',
                                        fontSize: '0.7rem',
                                        padding: '6px 8px',
                                        fontWeight: 'bold',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '6px'
                                    }}
                                >
                                    {isAnimationPaused ? '☀️ RESUME RIGGING' : '❄️ FREEZE AVATAR'}
                                </button>
                            </div>

                            {/* Sliders Container */}
                            <div style={{
                                flex: 1,
                                overflowY: 'auto',
                                paddingRight: '2px',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '12px'
                            }}>
                                {TARGET_BONES[activeRiggingGroup].map((bone: string) => (
                                    <div 
                                        key={bone} 
                                        style={{ 
                                            backgroundColor: 'rgba(255,255,255,0.01)', 
                                            border: '1px solid rgba(255,255,255,0.03)',
                                            borderRadius: '6px', 
                                            padding: '8px'
                                        }}
                                    >
                                        <div style={{ 
                                            fontSize: '0.75rem', 
                                            fontWeight: 'bold', 
                                            color: 'var(--dev-accent, #38BDF8)', 
                                            marginBottom: '8px',
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center'
                                        }}>
                                            <span>{bone}</span>
                                            <button 
                                                onClick={() => handleResetBone(bone)}
                                                style={{
                                                    background: 'none',
                                                    border: 'none',
                                                    color: '#6B7280',
                                                    cursor: 'pointer',
                                                    padding: '2px',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center'
                                                }}
                                                title={`Reset ${bone}`}
                                            >
                                                <RotateCcw size={10} className="hover:text-red-400 transition" />
                                            </button>
                                        </div>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                            {renderSlider(bone, 'x', '#EF4444')}
                                            {bone !== 'LeftUpLeg' && bone !== 'RightUpLeg' && bone !== 'LeftLeg' && bone !== 'RightLeg' && renderSlider(bone, 'y', '#10B981')}
                                            {renderSlider(bone, 'z', '#3B82F6')}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Rigging Global Footer Actions */}
                            <div style={{ 
                                display: 'flex', 
                                flexWrap: 'wrap', 
                                gap: '6px', 
                                borderTop: '1px solid rgba(255,255,255,0.05)', 
                                paddingTop: '8px',
                                flexShrink: 0
                            }}>
                                <button 
                                    onClick={exportSettings} 
                                    style={{
                                        ...styles.actionButton,
                                        flex: 1,
                                        backgroundColor: 'rgba(139, 92, 246, 0.15)',
                                        border: '1px solid rgba(139, 92, 246, 0.3)',
                                        color: '#C4B5FD',
                                        fontSize: '0.65rem',
                                        padding: '5px 4px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '4px'
                                    }}
                                >
                                    <Download size={10} />
                                    Export
                                </button>
                                <button 
                                    onClick={handleImportClick} 
                                    style={{
                                        ...styles.actionButton,
                                        flex: 1,
                                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                                        border: '1px solid rgba(16, 185, 129, 0.3)',
                                        color: '#A7F3D0',
                                        fontSize: '0.65rem',
                                        padding: '5px 4px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '4px'
                                    }}
                                >
                                    <FolderOpen size={10} />
                                    Import
                                </button>
                                <button 
                                    onClick={handleResetAll} 
                                    style={{
                                        ...styles.actionButton,
                                        flex: 1,
                                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                        border: '1px solid rgba(239, 68, 68, 0.2)',
                                        color: '#FCA5A5',
                                        fontSize: '0.65rem',
                                        padding: '5px 4px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '4px'
                                    }}
                                >
                                    <Trash2 size={10} />
                                    Reset All
                                </button>
                            </div>
                        </>
                    )}
                </div>

                {/* Hidden File Input for import */}
                <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={importSettings} 
                    accept=".json" 
                    style={{ display: 'none' }} 
                />
            </div>
        </div>
    );
};

export default PoseWorkshop;
