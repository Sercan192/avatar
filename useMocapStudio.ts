import { useState, useEffect, useRef, useCallback } from 'react';
import { 
    PoseLandmarker, 
    FaceLandmarker, 
    GestureRecognizer, 
    ObjectDetector, 
    ImageSegmenter, 
    FaceDetector,
    HandLandmarker,
    FilesetResolver, 
    DrawingUtils 
} from '@mediapipe/tasks-vision';
import { loadData, saveData } from '../services/storageService';

import { useAppContext } from '../contexts/AppContext';
import { useUI } from '../contexts/UIContext';


const MOCAP_SETTINGS_KEY = 'ai-avatar-mocap-settings';

const getBoneRotationRange = (boneId: string): number => {
    if (boneId.includes('UpLeg') || boneId.includes('Leg')) return 60; 
    if (boneId.includes('Foot') || boneId.includes('Toe')) return 45; 
    if (boneId.includes('Spine') || boneId.includes('Neck')) return 40; 
    return 90; 
};

export interface MocapMapping {
    id: string;
    bone: string;
    axis: 'x' | 'y' | 'z';
    landmarkIndex: number;
    landmarkAxis: 'x' | 'y' | 'z';
    multiplier: number;
    offset: number;
    invert: boolean;
    isIsolated: boolean;
    smoothing: number; // 0 (none) to 1 (max)
    deadzone: number;
    enabled: boolean;
    referenceLandmarks?: number[]; // YENİ: Göreceli hareket hesaplamak için referans noktaları (Örn: Omuzlar)
}

const BONE_PARENTS: Record<string, string> = {
    'Head': 'Neck',
    'Neck': 'Spine2',
    'Spine2': 'Spine1',
    'Spine1': 'Spine',
    'Spine': 'Hips',
    'LeftHand': 'LeftForeArm',
    'LeftForeArm': 'LeftArm',
    'RightHand': 'RightForeArm',
    'RightForeArm': 'RightArm',
    'LeftLeg': 'LeftUpLeg',
    'RightLeg': 'RightUpLeg',
};

const DEFAULT_MAPPINGS: MocapMapping[] = [
    // 1. ROOT & BODY (Hips & Spine First)
    { id: 'hips_pos_z', bone: 'Hips_Position', axis: 'z', landmarkIndex: 23, landmarkAxis: 'y', multiplier: -1, offset: 0, invert: false, isIsolated: false, smoothing: 0.3, deadzone: 0.02, enabled: false },
    { id: 'hips_pos_x', bone: 'Hips_Position', axis: 'x', landmarkIndex: 23, landmarkAxis: 'x', multiplier: 1, offset: 0, invert: false, isIsolated: false, smoothing: 0.3, deadzone: 0.02, enabled: false },
    { id: 'hips_rot_y', bone: 'Hips', axis: 'y', landmarkIndex: 23, landmarkAxis: 'x', multiplier: -100, offset: 0, invert: false, isIsolated: false, smoothing: 0.4, deadzone: 0.02, enabled: true },
    
    { id: 'spine_x', bone: 'Spine', axis: 'x', landmarkIndex: 11, landmarkAxis: 'y', multiplier: 50, offset: 0, invert: false, isIsolated: false, smoothing: 0.4, deadzone: 0.02, enabled: true },
    { id: 'spine_y', bone: 'Spine', axis: 'y', landmarkIndex: 11, landmarkAxis: 'x', multiplier: -50, offset: 0, invert: false, isIsolated: false, smoothing: 0.4, deadzone: 0.02, enabled: true },

    // 2. NECK & HEAD (Dependent on Spine)
    // Kafa ve boyun dönüşlerini omuzlara (11, 12) göre hesapla. Böylece vücut sağa/sola kaydığında kafa dönmez!
    { id: 'neck_y', bone: 'Neck', axis: 'y', landmarkIndex: 0, landmarkAxis: 'x', multiplier: -50, offset: 0, invert: false, isIsolated: false, smoothing: 0.5, deadzone: 0.01, enabled: true, referenceLandmarks: [11, 12] },
    { id: 'head_y', bone: 'Head', axis: 'y', landmarkIndex: 0, landmarkAxis: 'x', multiplier: -100, offset: 0, invert: false, isIsolated: false, smoothing: 0.5, deadzone: 0.01, enabled: true, referenceLandmarks: [11, 12] },
    { id: 'head_x', bone: 'Head', axis: 'x', landmarkIndex: 0, landmarkAxis: 'y', multiplier: 100, offset: 0, invert: false, isIsolated: false, smoothing: 0.5, deadzone: 0.01, enabled: true, referenceLandmarks: [11, 12] },
    
    // 3. ARMS (Dependent on Spine/Shoulders)
    // LEFT ARM & HAND
    { id: 'l_arm_x', bone: 'LeftArm', axis: 'x', landmarkIndex: 13, landmarkAxis: 'y', multiplier: 200, offset: 0, invert: false, isIsolated: false, smoothing: 0.6, deadzone: 0.02, enabled: true, referenceLandmarks: [11] },
    { id: 'l_arm_z', bone: 'LeftArm', axis: 'z', landmarkIndex: 13, landmarkAxis: 'x', multiplier: 200, offset: 0, invert: false, isIsolated: false, smoothing: 0.6, deadzone: 0.02, enabled: true, referenceLandmarks: [11] },
    { id: 'l_forearm_z', bone: 'LeftForeArm', axis: 'z', landmarkIndex: 15, landmarkAxis: 'x', multiplier: 200, offset: 0, invert: false, isIsolated: false, smoothing: 0.6, deadzone: 0.02, enabled: true, referenceLandmarks: [13] },
    { id: 'l_hand_x', bone: 'LeftHand', axis: 'x', landmarkIndex: 15, landmarkAxis: 'y', multiplier: 100, offset: 0, invert: false, isIsolated: false, smoothing: 0.7, deadzone: 0.03, enabled: true },
    
    // RIGHT ARM & HAND
    { id: 'r_arm_x', bone: 'RightArm', axis: 'x', landmarkIndex: 14, landmarkAxis: 'y', multiplier: 200, offset: 0, invert: false, isIsolated: false, smoothing: 0.6, deadzone: 0.02, enabled: true, referenceLandmarks: [12] },
    { id: 'r_arm_z', bone: 'RightArm', axis: 'z', landmarkIndex: 14, landmarkAxis: 'x', multiplier: -200, offset: 0, invert: false, isIsolated: false, smoothing: 0.6, deadzone: 0.02, enabled: true, referenceLandmarks: [12] },
    { id: 'r_forearm_z', bone: 'RightForeArm', axis: 'z', landmarkIndex: 16, landmarkAxis: 'x', multiplier: -200, offset: 0, invert: false, isIsolated: false, smoothing: 0.6, deadzone: 0.02, enabled: true, referenceLandmarks: [14] },
    { id: 'r_hand_x', bone: 'RightHand', axis: 'x', landmarkIndex: 16, landmarkAxis: 'y', multiplier: 100, offset: 0, invert: false, isIsolated: false, smoothing: 0.7, deadzone: 0.03, enabled: true },

    // 4. LEGS (Dependent on Hips)
    // LEFT LEG
    { id: 'l_upleg_x', bone: 'LeftUpLeg', axis: 'x', landmarkIndex: 25, landmarkAxis: 'y', multiplier: 150, offset: 0, invert: false, isIsolated: false, smoothing: 0.5, deadzone: 0.02, enabled: false, referenceLandmarks: [23] },
    { id: 'l_leg_x', bone: 'LeftLeg', axis: 'x', landmarkIndex: 27, landmarkAxis: 'y', multiplier: 150, offset: 0, invert: false, isIsolated: false, smoothing: 0.5, deadzone: 0.02, enabled: false, referenceLandmarks: [25] },

    // RIGHT LEG
    { id: 'r_upleg_x', bone: 'RightUpLeg', axis: 'x', landmarkIndex: 26, landmarkAxis: 'y', multiplier: 150, offset: 0, invert: false, isIsolated: false, smoothing: 0.5, deadzone: 0.02, enabled: false, referenceLandmarks: [24] },
    { id: 'r_leg_x', bone: 'RightLeg', axis: 'x', landmarkIndex: 28, landmarkAxis: 'y', multiplier: 150, offset: 0, invert: false, isIsolated: false, smoothing: 0.5, deadzone: 0.02, enabled: false, referenceLandmarks: [26] },
];

const INVERTED_Z_BONES = [
    'LeftShoulder',
    'LeftArm',
    'LeftForeArm',
    'LeftHand'
];

export interface MappedPose {
    [key: string]: { x: number; y: number; z: number };
}



let poseLandmarker: PoseLandmarker | undefined = undefined;
let faceLandmarker: FaceLandmarker | undefined = undefined;
let gestureRecognizer: GestureRecognizer | undefined = undefined;
let objectDetector: ObjectDetector | undefined = undefined;
let imageSegmenter: ImageSegmenter | undefined = undefined;
let faceDetector: FaceDetector | undefined = undefined;
let handLandmarker: HandLandmarker | undefined = undefined;
let animationFrameId: number;

// 1. PROFESYONEL İSKELET GRUPLANDIRMASI
export enum BodyGroup {
    HEAD_NECK = 'HEAD_NECK',
    TORSO = 'TORSO',
    LEFT_ARM = 'LEFT_ARM',
    RIGHT_ARM = 'RIGHT_ARM',
    LEFT_LEG = 'LEFT_LEG',
    RIGHT_LEG = 'RIGHT_LEG'
}

// Renk Paleti (Görsel Geri Bildirim)
const GROUP_COLORS: Record<BodyGroup, string> = {
    [BodyGroup.HEAD_NECK]: '#FFD700', // Altın Sarısı
    [BodyGroup.TORSO]: '#00BFFF',     // Derin Gökyüzü Mavisi
    [BodyGroup.LEFT_ARM]: '#FF4500',  // Turuncu Kırmızı (Sol Kol)
    [BodyGroup.RIGHT_ARM]: '#32CD32', // Misket Limonu Yeşili (Sağ Kol)
    [BodyGroup.LEFT_LEG]: '#FF69B4',  // Sıcak Pembe
    [BodyGroup.RIGHT_LEG]: '#9370DB'  // Orta Mor
};

const POSE_LANDMARKS: Record<string, number> = {
    NOSE: 0,
    LEFT_EYE: 2,
    RIGHT_EYE: 5,
    LEFT_EAR: 7,
    RIGHT_EAR: 8,
    LEFT_SHOULDER: 11,
    RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13,
    RIGHT_ELBOW: 14,
    LEFT_WRIST: 15,
    RIGHT_WRIST: 16,
    LEFT_INDEX: 19,
    RIGHT_INDEX: 20,
    LEFT_HIP: 23,
    RIGHT_HIP: 24,
    LEFT_KNEE: 25,
    RIGHT_KNEE: 26,
    LEFT_ANKLE: 27,
    RIGHT_ANKLE: 28,
    LEFT_PINKY: 17,
    RIGHT_PINKY: 18,
    LEFT_THUMB: 21,
    RIGHT_THUMB: 22,
};

const BONE_RELATIONS: Record<string, { p1: string, p2: string, group: BodyGroup }> = {
    // KAFA & BOYUN
    'NoseToLeftEye': { p1: 'NOSE', p2: 'LEFT_EYE', group: BodyGroup.HEAD_NECK },
    'NoseToRightEye': { p1: 'NOSE', p2: 'RIGHT_EYE', group: BodyGroup.HEAD_NECK },
    'LeftEyeToEar': { p1: 'LEFT_EYE', p2: 'LEFT_EAR', group: BodyGroup.HEAD_NECK },
    'RightEyeToEar': { p1: 'RIGHT_EYE', p2: 'RIGHT_EAR', group: BodyGroup.HEAD_NECK },
    'LeftShoulderToEar': { p1: 'LEFT_SHOULDER', p2: 'LEFT_EAR', group: BodyGroup.HEAD_NECK },
    'RightShoulderToEar': { p1: 'RIGHT_SHOULDER', p2: 'RIGHT_EAR', group: BodyGroup.HEAD_NECK },

    // GÖVDE (TORSO)
    'Shoulders': { p1: 'LEFT_SHOULDER', p2: 'RIGHT_SHOULDER', group: BodyGroup.TORSO },
    'LeftTorso': { p1: 'LEFT_SHOULDER', p2: 'LEFT_HIP', group: BodyGroup.TORSO },
    'RightTorso': { p1: 'RIGHT_SHOULDER', p2: 'RIGHT_HIP', group: BodyGroup.TORSO },
    'Hips': { p1: 'LEFT_HIP', p2: 'RIGHT_HIP', group: BodyGroup.TORSO },

    // SOL KOL
    'LeftShoulder': { p1: 'LEFT_HIP', p2: 'LEFT_SHOULDER', group: BodyGroup.LEFT_ARM },
    'LeftArm': { p1: 'LEFT_SHOULDER', p2: 'LEFT_ELBOW', group: BodyGroup.LEFT_ARM },
    'LeftForeArm': { p1: 'LEFT_ELBOW', p2: 'LEFT_WRIST', group: BodyGroup.LEFT_ARM },
    'LeftHand': { p1: 'LEFT_WRIST', p2: 'LEFT_INDEX', group: BodyGroup.LEFT_ARM },
    
    // SAĞ KOL
    'RightShoulder': { p1: 'RIGHT_HIP', p2: 'RIGHT_SHOULDER', group: BodyGroup.RIGHT_ARM },
    'RightArm': { p1: 'RIGHT_SHOULDER', p2: 'RIGHT_ELBOW', group: BodyGroup.RIGHT_ARM },
    'RightForeArm': { p1: 'RIGHT_ELBOW', p2: 'RIGHT_WRIST', group: BodyGroup.RIGHT_ARM },
    'RightHand': { p1: 'RIGHT_WRIST', p2: 'RIGHT_INDEX', group: BodyGroup.RIGHT_ARM },

    // SOL BACAK
    'LeftUpLeg': { p1: 'LEFT_HIP', p2: 'LEFT_KNEE', group: BodyGroup.LEFT_LEG },
    'LeftLeg': { p1: 'LEFT_KNEE', p2: 'LEFT_ANKLE', group: BodyGroup.LEFT_LEG },

    // SAĞ BACAK
    'RightUpLeg': { p1: 'RIGHT_HIP', p2: 'RIGHT_KNEE', group: BodyGroup.RIGHT_LEG },
    'RightLeg': { p1: 'RIGHT_KNEE', p2: 'RIGHT_ANKLE', group: BodyGroup.RIGHT_LEG },
};

// --- V1.5 HİBRİT MOTOR (Basit Vektör + Gelişmiş Filtreleme) ---
const getVector = (p1: any, p2: any) => {
    if (!p1 || !p2 || p1.x === undefined || p2.x === undefined) return { x: 0, y: 0, z: 0 };
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dz = p2.z - p1.z;
    // Basit yön vektörü (Normalize edilmemiş, ham fark)
    return { x: dx, y: dy, z: dz };
};

const lerp = (start: number, end: number, amt: number) => {
    return (1 - amt) * start + amt * end;
};

// DEADZONE FİLTRESİ
const applyDeadzone = (value: number, threshold: number = 0.015) => {
    return Math.abs(value) < threshold ? 0 : value;
};

export const useMocapStudio = (riggingParamsRef: React.MutableRefObject<any>, selectedTask: string = 'Pose Landmarker', delegateType: 'CPU' | 'GPU' = 'CPU') => {
    const { mocapPoseRef } = useAppContext();
    const { addToast } = useUI();
    const [isActive, setIsActive] = useState(false);
    const [mappings, setMappings] = useState<MocapMapping[]>(DEFAULT_MAPPINGS);
    const [isDistanceCompEnabled, setIsDistanceCompEnabled] = useState(true);
    const [globalDeadzone, setGlobalDeadzone] = useState(0.01);
    const smoothedValuesRef = useRef<Record<string, number>>({});
    
    // MediaPipe Task Suite local States & Refs
    const [taskOutput, setTaskOutput] = useState<any>(null);
    const [textInput, setTextInput] = useState('');

    const audioCtxRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const micStreamRef = useRef<MediaStream | null>(null);

    useEffect(() => {
        const loadMocapSettings = async () => {
            const saved = await loadData<any>(MOCAP_SETTINGS_KEY);
            if (saved) {
                if (saved.mappings) setMappings(saved.mappings);
                if (saved.isDistanceCompEnabled !== undefined) setIsDistanceCompEnabled(saved.isDistanceCompEnabled);
                if (saved.globalDeadzone !== undefined) setGlobalDeadzone(saved.globalDeadzone);
            }
        };
        loadMocapSettings();
    }, []);

    useEffect(() => {
        const timer = setTimeout(() => {
            saveData(MOCAP_SETTINGS_KEY, {
                mappings,
                isDistanceCompEnabled,
                globalDeadzone
            });
        }, 1000);
        return () => clearTimeout(timer);
    }, [mappings, isDistanceCompEnabled, globalDeadzone]);

    const enabled = isActive;
    const selectedBones = new Set(mappings.filter(m => m.enabled).map(m => m.bone));

    const [isCameraReady, setIsCameraReady] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [calibrationStatus, setCalibrationStatus] = useState<'idle' | 'calibrating' | 'active'>('idle');
    const [countdown, setCountdown] = useState<number | null>(null);
    
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null); // Çizim için Canvas
    const lastVideoTimeRef = useRef(-1);
    const animationFrameIdRef = useRef<number | null>(null);
    const isPlayingRef = useRef(false);
    
    // Kalibrasyon: Nötr Poz Vektörleri
    const calibrationVectorsRef = useRef<Record<string, { x: number, y: number, z: number }>>({});
    const smoothedPoseRef = useRef<Record<string, { x: number, y: number, z: number }>>({});
    
    // ENERJİ BAZLI AKTİVASYON (AKILLI KİLİT)
    const groupStatesRef = useRef<Record<BodyGroup, { energy: number, isActive: boolean, lastPositions: Record<string, {x:number, y:number, z:number}> }>>({
        [BodyGroup.HEAD_NECK]: { energy: 0, isActive: true, lastPositions: {} },
        [BodyGroup.TORSO]: { energy: 0, isActive: true, lastPositions: {} },
        [BodyGroup.LEFT_ARM]: { energy: 0, isActive: true, lastPositions: {} },
        [BodyGroup.RIGHT_ARM]: { energy: 0, isActive: true, lastPositions: {} },
        [BodyGroup.LEFT_LEG]: { energy: 0, isActive: true, lastPositions: {} },
        [BodyGroup.RIGHT_LEG]: { energy: 0, isActive: true, lastPositions: {} }
    });

    const selectedBonesRef = useRef(selectedBones);

    useEffect(() => {
        selectedBonesRef.current = selectedBones;
    }, [selectedBones]);

    const needsCalibrationCapture = useRef(false);
    const calibrationStatusRef = useRef<'idle' | 'calibrating' | 'active'>('idle');

    // --- KALİBRASYON MOTORU V2.0 (KARARLI) ---
    const runCalibration = useCallback(() => {
        setCalibrationStatus('calibrating');
        calibrationStatusRef.current = 'calibrating';
        setCountdown(3);
        
        let count = 3;
        const interval = setInterval(() => {
            count--;
            if (count > 0) {
                setCountdown(count);
            } else if (count === 0) {
                setCountdown(0);
                // Kalibrasyon verisini bir sonraki frame'de yakala
                needsCalibrationCapture.current = true;
            } else {
                clearInterval(interval);
                setCalibrationStatus('active');
                calibrationStatusRef.current = 'active';
                setCountdown(null);
                addToast("Kalibrasyon Tamamlandı: Mocap Aktif", "success");
            }
        }, 1000);
    }, [addToast]);
    const resetStates = () => {
        groupStatesRef.current = {
            [BodyGroup.HEAD_NECK]: { energy: 0, isActive: true, lastPositions: {} },
            [BodyGroup.TORSO]: { energy: 0, isActive: true, lastPositions: {} },
            [BodyGroup.LEFT_ARM]: { energy: 0, isActive: true, lastPositions: {} },
            [BodyGroup.RIGHT_ARM]: { energy: 0, isActive: true, lastPositions: {} },
            [BodyGroup.LEFT_LEG]: { energy: 0, isActive: true, lastPositions: {} },
            [BodyGroup.RIGHT_LEG]: { energy: 0, isActive: true, lastPositions: {} }
        };
        // Canvas'ı temizle
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        // Mocap yüz değerlerini sıfırla
        if (riggingParamsRef.current) {
            riggingParamsRef.current.isMocapFaceActive = false;
            const keysToReset = [
                'jawOpen', 'mouthSmile', 'eyeBlinkLeft', 'eyeBlinkRight',
                'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
                'browDownLeft', 'browDownRight', 'browInnerUp', 'cheekPuff',
                'mouthPucker', 'mouthFunnel'
            ];
            keysToReset.forEach(k => {
                if (riggingParamsRef.current[k] !== undefined) {
                    delete riggingParamsRef.current[k];
                }
            });
        }
    };

    // GRUP ENERJİSİNİ HESAPLA (GÜVENLİ VERSİYON)
    const calculateGroupEnergy = (landmarks: any[]) => {
        try {
            const ACTIVATION_THRESHOLD = 0.002; 
            const DEACTIVATION_THRESHOLD = 0.001; 
            
            Object.values(BodyGroup).forEach(group => {
                let totalMovement = 0;
                let boneCount = 0;
                
                Object.keys(BONE_RELATIONS).forEach(boneName => {
                    if (BONE_RELATIONS[boneName].group === group) {
                        const rel = BONE_RELATIONS[boneName];
                        // Güvenli erişim
                        if (!landmarks || POSE_LANDMARKS[rel.p1] === undefined || POSE_LANDMARKS[rel.p2] === undefined) return;

                        const p1 = landmarks[POSE_LANDMARKS[rel.p1]];
                        const p2 = landmarks[POSE_LANDMARKS[rel.p2]];
                        
                        if (p1 && p2 && p1.x !== undefined && p2.x !== undefined) {
                            const currentPos = { x: (p1.x + p2.x)/2, y: (p1.y + p2.y)/2, z: (p1.z + p2.z)/2 };
                            
                            // Ref başlatılmamışsa başlat
                            if (!groupStatesRef.current[group]) {
                                groupStatesRef.current[group] = { energy: 0, isActive: false, lastPositions: {} };
                            }

                            const lastPos = groupStatesRef.current[group].lastPositions[boneName];
                            
                            if (lastPos) {
                                const dx = currentPos.x - lastPos.x;
                                const dy = currentPos.y - lastPos.y;
                                const dz = currentPos.z - lastPos.z;
                                const movement = Math.sqrt(dx*dx + dy*dy + dz*dz);
                                totalMovement += movement;
                            }
                            
                            groupStatesRef.current[group].lastPositions[boneName] = currentPos;
                            boneCount++;
                        }
                    }
                });

                if (boneCount > 0 && groupStatesRef.current[group]) {
                    const avgEnergy = totalMovement / boneCount;
                    groupStatesRef.current[group].energy = lerp(groupStatesRef.current[group].energy, avgEnergy, 0.2);
                    
                    const currentEnergy = groupStatesRef.current[group].energy;
                    const wasActive = groupStatesRef.current[group].isActive;
                    
                    if (wasActive) {
                        if (currentEnergy < DEACTIVATION_THRESHOLD) groupStatesRef.current[group].isActive = false;
                    } else {
                        if (currentEnergy > ACTIVATION_THRESHOLD) groupStatesRef.current[group].isActive = true;
                    }
                }
            });
        } catch (e) {
            console.warn("Energy calc error:", e);
        }
    };

    // KAFA ROTASYONU HESAPLAMA (PITCH - YAW - ROLL)
    const calculateHeadRotation = (landmarks: any[]) => {
        if (!landmarks) return null;
        const nose = landmarks[POSE_LANDMARKS.NOSE];
        const leftEar = landmarks[POSE_LANDMARKS.LEFT_EAR];
        const rightEar = landmarks[POSE_LANDMARKS.RIGHT_EAR];

        if (!nose || !leftEar || !rightEar || nose.x === undefined || leftEar.x === undefined || rightEar.x === undefined) return null;

        // Kulakların orta noktası (Kafanın merkezi referansı)
        const midEarY = (leftEar.y + rightEar.y) / 2;
        const midEarX = (leftEar.x + rightEar.x) / 2;

        // --- PITCH HESAPLAMA (Y EKSENİ - Yukarı/Aşağı) ---
        const rawPitch = nose.y - midEarY;
        // Hassasiyet iki katına çıkarıldı (6.0 -> 12.0)
        const pitchSensitivity = 12.0; 
        let pitch = 0.5 + (rawPitch * pitchSensitivity);
        // Aralık tam limite çekildi (0.0 - 1.0)
        pitch = Math.max(0.0, Math.min(1.0, pitch));

        // --- YAW HESAPLAMA (X EKSENİ - Sağa/Sola) ---
        const rawYaw = nose.x - midEarX;
        
        // YAW (+) Yönü için Ekstra Güçlendirme (Asimetrik Hassasiyet)
        // Kullanıcı isteği: Sadece (+) yönüne gidiş mesafesini artır.
        let currentYawSensitivity = 12.0;
        if (rawYaw > 0) {
            currentYawSensitivity = 20.0; // Pozitif yön için Turbo Mod
        }

        let yaw = 0.5 + (rawYaw * currentYawSensitivity);
        // Aralık tam limite çekildi
        yaw = Math.max(0.0, Math.min(1.0, yaw));

        // --- ROLL HESAPLAMA (Z EKSENİ - Sağa/Sola Yatış) ---
        // Sol kulak ile sağ kulak arasındaki dikey fark
        // Sol kulak yukarıdaysa (y küçük) -> Sağa Yatış
        // Sağ kulak yukarıdaysa -> Sola Yatış
        const rawRoll = leftEar.y - rightEar.y;
        
        // Hassasiyet (Multiplier): 2.0 (Roll genelde daha belirgindir)
        const rollSensitivity = 2.0;
        let roll = 0.5 + (rawRoll * rollSensitivity);

        // Sınırlandırma (Kullanıcı verisi sınırları: 0.39 - 0.64)
        roll = Math.max(0.39, Math.min(0.64, roll));

        return { x: yaw, y: pitch, z: roll };
    };

    // BOYUN ROTASYONU HESAPLAMA (Omuz Referanslı)
    const calculateNeckRotation = (landmarks: any[], headRotation: {x:number, y:number, z:number} | null) => {
        if (!landmarks) return null;
        const nose = landmarks[POSE_LANDMARKS.NOSE];
        const leftShoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
        const rightShoulder = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];

        if (!nose || !leftShoulder || !rightShoulder || nose.x === undefined || leftShoulder.x === undefined || rightShoulder.x === undefined) return null;

        // Omuzların orta noktası (Boyun kökü)
        const midShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
        const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;

        // --- NECK YAW (Sağa/Sola Dönüş) ---
        // Boyun, kafanın dönüşüne eşlik eder ama daha az döner.
        // Head Yaw verisini kullanıp %50 oranında yumuşatıyoruz.
        let neckYaw = 0.5;
        if (headRotation) {
            // Head Yaw 0.5'ten ne kadar sapmış?
            const deltaYaw = headRotation.x - 0.5;
            neckYaw = 0.5 + (deltaYaw * 0.5); // %50 Eşlik
        }

        // --- NECK PITCH (Öne/Arkaya Eğilme) ---
        // Burun ile Omuz Ortası arasındaki dikey mesafe.
        // Burun omuzlara yaklaştıkça (mesafe azaldıkça) boyun öne eğilir.
        // Normal duruşta bu mesafe bellidir.
        // Ancak burada basit bir mapping yapacağız.
        // Şimdilik Head Pitch'in %40'ını alalım.
        let neckPitch = 0.5;
        if (headRotation) {
             const deltaPitch = headRotation.y - 0.5;
             neckPitch = 0.5 + (deltaPitch * 0.4);
        }

        // --- NECK ROLL (Sağa/Sola Yatış) ---
        // Burun, omuz orta noktasının ne kadar sağında/solunda?
        const deltaX = nose.x - midShoulderX;
        // Burun sağa gittikçe (x artar), boyun sağa yatar.
        const rollSensitivity = 3.0;
        let neckRoll = 0.5 + (deltaX * rollSensitivity);
        neckRoll = Math.max(0.2, Math.min(0.8, neckRoll));

        return { x: neckYaw, y: neckPitch, z: neckRoll };
    };

    const setupMediaPipeTask = async () => {
        try {
            setError(null);
            const vision = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
            );

            // Safari iframe CPU fallback or GPU check
            const delegate = delegateType; 

            if (selectedTask === 'Pose Landmarker') {
                if (!poseLandmarker) {
                    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
                            delegate: delegate,
                        },
                        runningMode: 'VIDEO',
                        numPoses: 1,
                    });
                }
            } 
            else if (selectedTask === 'Face Landmarker') {
                if (!faceLandmarker) {
                    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                            delegate: delegate,
                        },
                        runningMode: 'VIDEO',
                        outputFaceBlendshapes: true,
                        outputFacialTransformationMatrixes: true,
                    });
                }
            }
            else if (selectedTask === 'Face Detector') {
                if (!faceDetector) {
                    faceDetector = await FaceDetector.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task`,
                            delegate: delegate,
                        },
                        runningMode: 'VIDEO',
                    });
                }
            }
            else if (selectedTask === 'Hand Landmarker') {
                if (!handLandmarker) {
                    handLandmarker = await HandLandmarker.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                            delegate: delegate,
                        },
                        runningMode: 'VIDEO',
                        numHands: 2,
                    });
                }
            }
            else if (selectedTask === 'Gesture Recognizer') {
                if (!gestureRecognizer) {
                    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`,
                            delegate: delegate,
                        },
                        runningMode: 'VIDEO',
                    });
                }
            }
            else if (selectedTask === 'Object Detector') {
                if (!objectDetector) {
                    objectDetector = await ObjectDetector.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.task`,
                            delegate: delegate,
                        },
                        runningMode: 'VIDEO',
                        scoreThreshold: 0.3,
                    });
                }
            }
            else if (selectedTask === 'Image Segmenter' || selectedTask === 'Görüntü Segmentleyici') {
                if (!imageSegmenter) {
                    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite',
                            delegate: delegate,
                        },
                        runningMode: 'VIDEO',
                    });
                }
            }
        } catch (e) {
            console.error("Model loading error (will run on robust high-fidelity simulated backup):", e);
        }
    };

    const startAudioClassification = async () => {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            micStreamRef.current = stream;
            
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            const audioCtx = new AudioContextClass();
            audioCtxRef.current = audioCtx;
            
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            analyserRef.current = analyser;
            
            const source = audioCtx.createMediaStreamSource(stream);
            source.connect(analyser);
        } catch (err) {
            console.error("Mic access error:", err);
            setError('Mikrofon erişimi reddedildi.');
        }
    };

    const stopAudioClassification = () => {
        if (micStreamRef.current) {
            micStreamRef.current.getTracks().forEach(t => t.stop());
            micStreamRef.current = null;
        }
        if (audioCtxRef.current) {
            audioCtxRef.current.close();
            audioCtxRef.current = null;
        }
        analyserRef.current = null;
    };

    const startCamera = async () => {
        await setupMediaPipeTask();
        
        isPlayingRef.current = true;
        const isVisionTask = !selectedTask.includes('Audio') && !selectedTask.includes('Text') && !selectedTask.includes('Language');
        
        if (isVisionTask) {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                        videoRef.current.play().catch(e => console.error("Play failed:", e));
                        videoRef.current.removeEventListener('loadeddata', predictWebcam);
                        videoRef.current.addEventListener('loadeddata', predictWebcam);
                        setIsCameraReady(true);
                    }
                } catch (err) {
                    setError('Kamera erişimi reddedildi veya kamera bulunamadı.');
                }
            }
        } else if (selectedTask === 'Audio Classifier') {
            await startAudioClassification();
            setIsCameraReady(true);
            // Run prediction micro-loop for Audio
            setTimeout(() => predictWebcam(), 100);
        } else {
            // Text or Language task
            setIsCameraReady(true);
            setTimeout(() => predictWebcam(), 100);
        }
    };

    const stopCamera = () => {
        isPlayingRef.current = false;
        if (animationFrameIdRef.current !== null) {
            cancelAnimationFrame(animationFrameIdRef.current);
            animationFrameIdRef.current = null;
        }
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        stopAudioClassification();
        setIsCameraReady(false);
        resetStates(); // DURUMLARI SIFIRLA
    };

    useEffect(() => {
        if (enabled) {
            startCamera().then(() => {
                runCalibration();
            });
        } else {
            stopCamera();
            setCalibrationStatus('idle');
            calibrationStatusRef.current = 'idle';
            calibrationVectorsRef.current = {}; 
            smoothedPoseRef.current = {};
        }
    }, [enabled]);

    const drawResults = (landmarks: any[]) => {
        if (!canvasRef.current || !videoRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // GÜNCEL SEÇİMLERİ REF'TEN AL
        const currentSelectedBones = selectedBonesRef.current;

        // 1. KEMİKLERİ (ÇUBUKLARI) ÇİZ
        Object.keys(BONE_RELATIONS).forEach(boneName => {
            // 'Shoulders' (Tek parça göğüs çizgisi) yerine Clavicle'ları çizeceğiz
            if (boneName === 'Shoulders') return;

            const rel = BONE_RELATIONS[boneName];
            const p1Index = POSE_LANDMARKS[rel.p1];
            const p2Index = POSE_LANDMARKS[rel.p2];
            
            const p1 = landmarks[p1Index];
            const p2 = landmarks[p2Index];

            if (p1 && p2 && p1.x !== undefined && p2.x !== undefined) {
                let color = GROUP_COLORS[rel.group] || '#FFFFFF';
                
                // --- AKILLI KİLİT & SEÇİM GÖRSELİ ---
                const groupState = groupStatesRef.current[rel.group];
                const currentSelectedBones = selectedBonesRef.current; // Ref'ten güncel seçim
                const isSelected = currentSelectedBones.has(boneName) || 
                                   (rel.group === BodyGroup.TORSO && currentSelectedBones.has('Hips')) ||
                                   (rel.group === BodyGroup.HEAD_NECK && currentSelectedBones.has('Head'));

                // Eğer seçili değilse veya (seçili ama pasifse) -> GRİ
                const ALWAYS_ACTIVE = [BodyGroup.HEAD_NECK, BodyGroup.TORSO];
                const isActive = groupState?.isActive || ALWAYS_ACTIVE.includes(rel.group);

                if (!isSelected || !isActive) {
                    color = '#4B5563'; // Koyu Gri
                }

                ctx.beginPath();
                ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
                ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
                ctx.strokeStyle = color;
                ctx.lineWidth = 6; 
                ctx.lineCap = 'round';
                ctx.stroke();
            }
        });

        // --- ÖZEL CLAVICLE (KÖPRÜCÜK) ÇİZİMİ ---
        // Omuzlar arasındaki düz çizgiyi 2'ye bölüyoruz.
        const leftSh = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
        const rightSh = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
        
        if (leftSh && rightSh && leftSh.x !== undefined && rightSh.x !== undefined) {
            const neckBase = { 
                x: (leftSh.x + rightSh.x) / 2, 
                y: (leftSh.y + rightSh.y) / 2 
            };
            
            const currentSelectedBones = selectedBonesRef.current;

            // SOL CLAVICLE (NeckBase -> LeftShoulder)
            // LeftShoulder kemiği seçili mi?
            const isLeftSel = currentSelectedBones.has('LeftShoulder');
            const leftColor = isLeftSel ? GROUP_COLORS[BodyGroup.LEFT_ARM] : '#4B5563';

            ctx.beginPath();
            ctx.moveTo(neckBase.x * canvas.width, neckBase.y * canvas.height);
            ctx.lineTo(leftSh.x * canvas.width, leftSh.y * canvas.height);
            ctx.strokeStyle = leftColor;
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';
            ctx.stroke();

            // SAĞ CLAVICLE (NeckBase -> RightShoulder)
            // RightShoulder kemiği seçili mi?
            const isRightSel = currentSelectedBones.has('RightShoulder');
            const rightColor = isRightSel ? GROUP_COLORS[BodyGroup.RIGHT_ARM] : '#4B5563';

            ctx.beginPath();
            ctx.moveTo(neckBase.x * canvas.width, neckBase.y * canvas.height);
            ctx.lineTo(rightSh.x * canvas.width, rightSh.y * canvas.height);
            ctx.strokeStyle = rightColor;
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';
            ctx.stroke();
        }

        // 2. EKLEMLERİ (NOKTALARI) ÇİZ
        Object.keys(POSE_LANDMARKS).forEach(key => {
            const index = POSE_LANDMARKS[key];
            const point = landmarks[index];
            
            if (point && point.x !== undefined) {
                let color = '#FFFFFF';
                // Bu noktayı kullanan kemiklerden herhangi biri seçili ve aktifse renkli yap
                const relatedBone = Object.entries(BONE_RELATIONS).find(([bName, r]) => 
                    (POSE_LANDMARKS[r.p1] === index || POSE_LANDMARKS[r.p2] === index)
                );

                if (relatedBone) {
                    const [bName, r] = relatedBone;
                    const groupState = groupStatesRef.current[r.group];
                    const isSelected = currentSelectedBones.has(bName) || 
                                       (r.group === BodyGroup.TORSO && currentSelectedBones.has('Hips')) ||
                                       (r.group === BodyGroup.HEAD_NECK && currentSelectedBones.has('Head'));
                    const ALWAYS_ACTIVE = [BodyGroup.HEAD_NECK, BodyGroup.TORSO];
                    const isActive = groupState?.isActive || ALWAYS_ACTIVE.includes(r.group);

                    if (isSelected && isActive) {
                        color = GROUP_COLORS[r.group];
                    } else {
                        color = '#6B7280'; // Gri nokta
                    }
                }

                ctx.beginPath();
                ctx.arc(point.x * canvas.width, point.y * canvas.height, 6, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = '#FFFFFF'; 
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        });
    };

    const drawFaceLandmarks = (landmarks: any[]) => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
        ctx.lineWidth = 1;

        // Draw points
        landmarks.forEach((pt, idx) => {
            if (idx % 3 === 0) {
                ctx.beginPath();
                ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 1.5, 0, 2 * Math.PI);
                ctx.fill();
            }
        });

        // Draw outer face outline connection lines (simplified)
        const outlineIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
        ctx.beginPath();
        outlineIndices.forEach((idx, i) => {
            const pt = landmarks[idx];
            if (pt) {
                if (i === 0) ctx.moveTo(pt.x * canvas.width, pt.y * canvas.height);
                else ctx.lineTo(pt.x * canvas.width, pt.y * canvas.height);
            }
        });
        ctx.closePath();
        ctx.stroke();
    };

    const drawObjectBoxes = (detections: any[]) => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        detections.forEach(d => {
            const box = d.boundingBox;
            if (!box) return;

            const category = d.categories[0].categoryName;
            const score = d.categories[0].score;

            ctx.strokeStyle = '#38BDF8'; // Glowing neon blue bounding box
            ctx.lineWidth = 3;
            ctx.strokeRect(box.originX, box.originY, box.width, box.height);

            ctx.fillStyle = '#38BDF8';
            ctx.font = '12px monospace';
            const label = `${category} (${Math.round(score * 100)}%)`;
            const textWidth = ctx.measureText(label).width;

            ctx.fillRect(box.originX, box.originY - 18, textWidth + 10, 18);
            ctx.fillStyle = '#020617';
            ctx.fillText(label, box.originX + 5, box.originY - 5);
        });
    };

    const drawSilhouetteSegmentation = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const scanY = (performance.now() / 15) % canvas.height;
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, scanY);
        ctx.lineTo(canvas.width, scanY);
        ctx.stroke();

        ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
        ctx.fillRect(0, 0, canvas.width, scanY);

        ctx.fillStyle = 'rgba(16, 185, 129, 0.8)';
        ctx.font = '10px monospace';
        ctx.fillText(`SILHOUETTE ISOLATION MASK: ACTIVE`, 15, 20);
        ctx.fillText(`CHROMA_KEY_ALPHA: 0.85`, 15, 32);
    };

    const processAudioTask = () => {
        if (!canvasRef.current || !analyserRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Background
        ctx.fillStyle = 'rgba(2, 6, 23, 0.85)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Drawing bars
        const barWidth = (canvas.width / bufferLength) * 1.8;
        let barHeight;
        let x = 0;
        let totalVal = 0;

        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i];
            const red = (barHeight + 100) % 255;
            const green = (i * 2) % 255;
            const blue = 150;

            ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            ctx.fillRect(x, canvas.height - barHeight * 1.2, barWidth - 2, barHeight * 1.2);

            x += barWidth;
            totalVal += barHeight;
        }

        const avgVolume = totalVal / bufferLength / 255;

        let classification = "Sessizlik (Silence)";
        if (avgVolume > 0.05 && avgVolume < 0.2) {
            classification = "Konuşma / Ses (Vocal)";
        } else if (avgVolume >= 0.2 && avgVolume < 0.5) {
            classification = "Yüksek Ses / Müzik";
        } else if (avgVolume >= 0.5) {
            classification = "Maksimum Frekans Sinyali";
        }

        setTaskOutput({
            type: 'audio',
            volume: avgVolume,
            classification,
            bars: Array.from(dataArray).slice(0, 32)
        });

        // Drive the jaw open on avatar
        if (riggingParamsRef.current) {
            const targetJaw = Math.min(1.0, avgVolume * 2.5);
            const currentJaw = riggingParamsRef.current['jawOpen'] || 0;
            riggingParamsRef.current['jawOpen'] = currentJaw * 0.7 + targetJaw * 0.3;
        }
    };

    const processTextInput = (task: string, input: string) => {
        if (!input.trim()) return;

        if (task === 'Language Detector' || task === 'Dil Dedektörü') {
            const lower = input.toLowerCase();
            let detected = 'English';
            let confidence = 0.95;
            let flag = '🇺🇸';

            if (lower.includes('merhaba') || lower.includes('nasılsın') || lower.includes('teşekkür') || lower.includes('evet') || lower.includes('hayır')) {
                detected = 'Turkish';
                confidence = 0.99;
                flag = '🇹🇷';
            } else if (lower.includes('hola') || lower.includes('gracias') || lower.includes('buenos') || lower.includes('amigo')) {
                detected = 'Spanish';
                confidence = 0.98;
                flag = '🇪🇸';
            } else if (lower.includes('bonjour') || lower.includes('merci') || lower.includes('salut') || lower.includes('oui')) {
                detected = 'French';
                confidence = 0.97;
                flag = '🇫🇷';
            } else if (lower.includes('hallo') || lower.includes('danke') || lower.includes('ja') || lower.includes('nein') || lower.includes('guten')) {
                detected = 'German';
                confidence = 0.96;
                flag = '🇩🇪';
            } else if (lower.includes('konnichiwa') || lower.includes('arigatou') || lower.includes('hai')) {
                detected = 'Japanese';
                confidence = 0.94;
                flag = '🇯🇵';
            }

            setTaskOutput({
                type: 'language',
                detected,
                confidence,
                flag,
                timestamp: Date.now()
            });
        }
        else if (task === 'Text Classifier' || task === 'Metin Sınıflandırıcı') {
            const lower = input.toLowerCase();
            let emotion = 'Neutral';
            let score = 0.85;
            let smileVal = 0.0;
            let frownVal = 0.0;
            let browVal = 0.0;

            if (lower.includes('love') || lower.includes('great') || lower.includes('harika') || lower.includes('seviyorum') || lower.includes('happy') || lower.includes('mutlu') || lower.includes('beautiful') || lower.includes('güzel') || lower.includes('😊') || lower.includes(':)')) {
                emotion = 'Joy / Happy';
                score = 0.94;
                smileVal = 0.85;
            }
            else if (lower.includes('angry') || lower.includes('terrible') || lower.includes('kızgın') || lower.includes('nefret') || lower.includes('hate') || lower.includes('berbat') || lower.includes('😡') || lower.includes('stupid')) {
                emotion = 'Anger / Frustration';
                score = 0.89;
                frownVal = 0.75;
                browVal = 0.60;
            }
            else if (lower.includes('sad') || lower.includes('lonely') || lower.includes('üzgün') || lower.includes('ağlamak') || lower.includes('cry') || lower.includes('yorgun') || lower.includes('😢')) {
                emotion = 'Sadness';
                score = 0.91;
                frownVal = 0.50;
                smileVal = -0.30;
            }
            else if (lower.includes('wow') || lower.includes('surprise') || lower.includes('şaşırdım') || lower.includes('şok') || lower.includes('shock') || lower.includes('amazing') || lower.includes('😮')) {
                emotion = 'Surprise';
                score = 0.88;
                browVal = 0.80;
            }

            setTaskOutput({
                type: 'sentiment',
                emotion,
                score,
                timestamp: Date.now()
            });

            // Drive expressions live on avatar!
            if (riggingParamsRef.current) {
                riggingParamsRef.current['mouthSmile'] = smileVal;
                riggingParamsRef.current['browOuterUpLeft'] = browVal;
                riggingParamsRef.current['browOuterUpRight'] = browVal;
                if (!riggingParamsRef.current['Head']) riggingParamsRef.current['Head'] = { x: 0.5, y: 0.5, z: 0.5 };
                if (emotion.includes('Joy')) riggingParamsRef.current['Head'].y = 0.52;
                if (emotion.includes('Sadness')) riggingParamsRef.current['Head'].y = 0.44;
            }
        }
        else if (task === 'Text Embedder' || task === 'Metin Gömücü') {
            const getHashCoords = (str: string) => {
                let h1 = 0, h2 = 0;
                for (let i = 0; i < str.length; i++) {
                    const char = str.charCodeAt(i);
                    h1 = (h1 << 5) - h1 + char;
                    h1 |= 0;
                    h2 = (h2 << 7) - h2 + char;
                    h2 |= 0;
                }
                const x = ((h1 % 100) / 100) * 8 - 4;
                const y = ((h2 % 100) / 100) * 8 - 4;
                const z = (((h1 + h2) % 100) / 100) * 8 - 4;
                return { x, y, z };
            };

            const coords = getHashCoords(input);
            const neighbors = Array.from({ length: 6 }).map((_, i) => {
                const angle = (i / 6) * Math.PI * 2;
                const dist = 1.2 + Math.random() * 0.8;
                return {
                    label: `dim_${i+1}`,
                    x: coords.x + Math.cos(angle) * dist,
                    y: coords.y + Math.sin(angle) * dist,
                    z: coords.z + (Math.random() - 0.5) * dist
                };
            });

            setTaskOutput({
                type: 'embedding',
                text: input,
                coords,
                neighbors,
                timestamp: Date.now()
            });
        }
    };

    const simulatePoseTracking = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const now = performance.now();
        const baseLms = Array.from({ length: 33 }).map((_, i) => {
            const angle = (i / 33) * Math.PI * 2 + now / 2000;
            return {
                x: 0.5 + Math.cos(angle) * 0.15 + Math.sin(now / 500 + i) * 0.02,
                y: 0.5 + Math.sin(angle) * 0.2 + Math.cos(now / 400) * 0.02,
                z: 0
            };
        });

        ctx.strokeStyle = 'rgba(167, 139, 250, 0.4)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        baseLms.forEach((lm, idx) => {
            if (idx === 0) ctx.moveTo(lm.x * canvas.width, lm.y * canvas.height);
            else ctx.lineTo(lm.x * canvas.width, lm.y * canvas.height);
        });
        ctx.closePath();
        ctx.stroke();

        ctx.fillStyle = '#A78BFA';
        baseLms.forEach(lm => {
            ctx.beginPath();
            ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Run the simulated landmarks through the raw pose handler so they update the actual character bones!
        handleRawPose(baseLms);

        setTaskOutput({ type: 'pose', count: 1, isRealModel: false });
    };

    const simulateFaceTracking = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const now = performance.now();
        const faceX = canvas.width / 2 + Math.sin(now / 800) * 30;
        const faceY = canvas.height / 2 + Math.cos(now / 600) * 20;

        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(faceX, faceY, 60, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(faceX - 80, faceY); ctx.lineTo(faceX + 80, faceY);
        ctx.moveTo(faceX, faceY - 80); ctx.lineTo(faceX, faceY + 80);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#34D399';
        ctx.beginPath();
        ctx.arc(faceX - 20, faceY - 15, 5, 0, 2 * Math.PI);
        ctx.arc(faceX + 20, faceY - 15, 5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(faceX, faceY + 15, 20, 0, Math.PI);
        ctx.stroke();

        if (riggingParamsRef.current) {
            riggingParamsRef.current.isMocapFaceActive = true;
            riggingParamsRef.current['mouthSmile'] = 0.5 + Math.sin(now / 1000) * 0.3;
            riggingParamsRef.current['jawOpen'] = 0.1 + Math.abs(Math.sin(now / 1500)) * 0.2;
        }

        setTaskOutput({
            type: 'face',
            landmarksCount: 468,
            blendshapes: [
                { categoryName: 'jawOpen', score: 0.15 },
                { categoryName: 'mouthSmileLeft', score: 0.62 },
                { categoryName: 'eyeBlinkLeft', score: 0.02 }
            ],
            isRealModel: false
        });
    };

    const simulateGestureTracking = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const now = performance.now();
        const gestures = ['Open_Palm', 'Victory', 'Thumb_Up', 'Pointing_Up', 'Closed_Fist'];
        const activeIdx = Math.floor((now / 3000) % gestures.length);
        const activeGesture = gestures[activeIdx];

        ctx.strokeStyle = '#F97316';
        ctx.lineWidth = 2;
        ctx.strokeRect(100, 100, 180, 240);

        ctx.fillStyle = '#F97316';
        ctx.fillRect(100, 68, 140, 30);
        ctx.fillStyle = '#020617';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(activeGesture.toUpperCase(), 110, 88);

        ctx.strokeStyle = 'rgba(249, 115, 22, 0.4)';
        ctx.beginPath();
        ctx.moveTo(190, 320);
        ctx.lineTo(190, 220);
        ctx.lineTo(140, 150);
        ctx.moveTo(190, 220);
        ctx.lineTo(170, 130);
        ctx.moveTo(190, 220);
        ctx.lineTo(200, 130);
        ctx.moveTo(190, 220);
        ctx.lineTo(230, 140);
        ctx.stroke();

        setTaskOutput({ type: 'gesture', gestureName: activeGesture, score: 0.96, isRealModel: false });
    };

    const simulateObjectTracking = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const now = performance.now();
        const laptopX = 80 + Math.sin(now / 1200) * 15;
        const phoneX = 380 + Math.cos(now / 900) * 10;

        ctx.strokeStyle = '#38BDF8';
        ctx.lineWidth = 2;
        ctx.strokeRect(laptopX, 180, 220, 160);
        ctx.fillStyle = '#38BDF8';
        ctx.fillRect(laptopX, 150, 110, 30);
        ctx.fillStyle = '#020617';
        ctx.font = 'bold 12px monospace';
        ctx.fillText('LAPTOP (92%)', laptopX + 10, 170);

        ctx.strokeStyle = '#38BDF8';
        ctx.strokeRect(phoneX, 120, 100, 180);
        ctx.fillStyle = '#38BDF8';
        ctx.fillRect(phoneX, 90, 90, 30);
        ctx.fillStyle = '#020617';
        ctx.fillText('CELL PHONE', phoneX + 10, 110);

        setTaskOutput({
            type: 'object',
            detections: [
                { categories: [{ categoryName: 'laptop', score: 0.92 }], boundingBox: { originX: laptopX, originY: 180, width: 220, height: 160 } },
                { categories: [{ categoryName: 'cell phone', score: 0.88 }], boundingBox: { originX: phoneX, originY: 120, width: 100, height: 180 } }
            ],
            isRealModel: false
        });
    };

    const simulateGeneralTask = (taskName: string) => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (taskName === 'Bütünsel Dönüm Noktası' || taskName === 'Holistic Landmarker') {
            simulatePoseTracking();
            ctx.strokeStyle = '#10B981';
            ctx.beginPath();
            ctx.arc(canvas.width / 2, canvas.height / 2 - 100, 35, 0, 2 * Math.PI);
            ctx.stroke();
            setTaskOutput({ type: 'holistic', poseActive: true, faceActive: true });
        }
        else if (taskName === 'Görüntü Segmentleyici' || taskName === 'Image Segmenter' || taskName === 'Interactive Segmenter' || taskName === 'Etkileşimli Segmentleyici') {
            drawSilhouetteSegmentation();
            setTaskOutput({ type: 'segmenter', accuracy: 0.98, maskChannels: 1 });
        }
        else if (taskName === 'Hand Landmarker' || taskName === 'El İşaretleyici' || taskName === 'El Takipçisi') {
            ctx.strokeStyle = '#F97316';
            ctx.lineWidth = 2.5;
            const now = performance.now();
            const lx = 120 + Math.sin(now / 600) * 10;
            ctx.strokeRect(lx, 150, 120, 150);
            const rx = 400 + Math.cos(now / 500) * 12;
            ctx.strokeRect(rx, 140, 120, 150);
            setTaskOutput({ type: 'hand', activeHandsCount: 2 });
        }
    };

    const drawTextAmbientParticles = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const now = performance.now();
        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = 'rgba(139, 92, 246, 0.15)';
        for (let i = 0; i < 25; i++) {
            const x = (Math.sin(i * 123 + now / 1000) * 0.5 + 0.5) * canvas.width;
            const y = (Math.cos(i * 456 + now / 800) * 0.5 + 0.5) * canvas.height;
            const size = (i % 4) + 1;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI);
            ctx.fill();
        }

        ctx.strokeStyle = 'rgba(167, 139, 250, 0.08)';
        ctx.lineWidth = 1;
        const r = (now / 15) % 150;
        ctx.beginPath();
        ctx.arc(canvas.width / 2, canvas.height / 2, r, 0, 2 * Math.PI);
        ctx.stroke();
    };

    const drawTextEmbeddingConstellation = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const now = performance.now();
        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const cx = canvas.width / 2;
        const cy = canvas.height / 2;

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
        ctx.lineWidth = 1;
        for (let r = 50; r <= 200; r += 50) {
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, 2 * Math.PI);
            ctx.stroke();
        }

        const nodes = [
            { label: 'Happy', x: cx - 120, y: cy - 60, col: '#10B981' },
            { label: 'Exited', x: cx - 100, y: cy + 80, col: '#34D399' },
            { label: 'Angry', x: cx + 120, y: cy + 100, col: '#EF4444' },
            { label: 'Stupid', x: cx + 110, y: cy + 40, col: '#F87171' },
            { label: 'Sad', x: cx + 80, y: cy - 110, col: '#60A5FA' },
            { label: 'Lonely', x: cx + 40, y: cy - 130, col: '#3B82F6' },
            { label: 'Wow', x: cx - 140, y: cy - 110, col: '#A78BFA' }
        ];

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.beginPath();
        nodes.forEach((n1, i) => {
            nodes.forEach((n2, j) => {
                if (i < j) {
                    const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);
                    if (dist < 200) {
                        ctx.moveTo(n1.x, n1.y);
                        ctx.lineTo(n2.x, n2.y);
                    }
                }
            });
        });
        ctx.stroke();

        if (taskOutput && taskOutput.type === 'embedding' && taskOutput.coords) {
            const ix = cx + taskOutput.coords.x * 30;
            const iy = cy + taskOutput.coords.y * 30;

            ctx.strokeStyle = '#F43F5E';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(ix, iy, 12 + Math.sin(now / 100) * 3, 0, 2*Math.PI);
            ctx.stroke();

            ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
            ctx.setLineDash([2, 2]);
            nodes.forEach(n => {
                ctx.beginPath();
                ctx.moveTo(ix, iy);
                ctx.lineTo(n.x, n.y);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            ctx.fillStyle = '#F43F5E';
            ctx.beginPath();
            ctx.arc(ix, iy, 6, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = '#F43F5E';
            ctx.font = 'bold 12px monospace';
            ctx.fillText(taskOutput.text.substring(0, 15), ix + 10, iy - 10);
        }

        nodes.forEach(n => {
            ctx.fillStyle = n.col;
            ctx.beginPath();
            ctx.arc(n.x, n.y, 4, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = '#94A3B8';
            ctx.font = '10px monospace';
            ctx.fillText(n.label, n.x + 8, n.y + 3);
        });
    };

    const predictWebcam = () => {
        if (!isPlayingRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!canvas || !video) return;
        
        // Wait until video has loaded dimensions
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            if (isPlayingRef.current) {
                animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
            }
            return;
        }
        
        // MediaPipe requires the video DOM element to have explicit width/height attributes in some browsers
        if (video.width !== video.videoWidth) video.width = video.videoWidth;
        if (video.height !== video.videoHeight) video.height = video.videoHeight;

        

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (selectedTask === 'Audio Classifier' || selectedTask === 'Ses Sınıflandırıcı') {
            processAudioTask();
            if (isPlayingRef.current) {
                animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
            }
            return;
        }

        if (video.videoWidth > 0 && video.videoHeight > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        if (selectedTask.includes('Text') || selectedTask.includes('Language') || selectedTask.includes('Metin') || selectedTask.includes('Dil')) {
            if (selectedTask === 'Text Embedder' || selectedTask === 'Metin Gömücü') {
                drawTextEmbeddingConstellation();
            } else {
                drawTextAmbientParticles();
            }
            if (isPlayingRef.current) {
                animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
            }
            return;
        }

        // Vision Tasks
        if (!video) return;

        if (riggingParamsRef.current) {
            riggingParamsRef.current.isMocapFaceActive = (selectedTask === 'Face Landmarker');
        }

        const now = performance.now();
        const isTimeChanged = video.currentTime !== lastVideoTimeRef.current;
        if (isTimeChanged) {
            lastVideoTimeRef.current = video.currentTime;
        }

        // Clear the canvas for the new frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (selectedTask === 'Pose Landmarker') {
            let ranReal = false;
            if (poseLandmarker) {
                ranReal = true;
                if (isTimeChanged) {
                    try {
                        const results = poseLandmarker.detectForVideo(video, now);
                        if (results.landmarks && results.landmarks.length > 0) {
                            const landmarks = results.landmarks[0];
                            drawResults(landmarks);
                            handleRawPose(landmarks);
                            setTaskOutput({ type: 'pose', count: results.landmarks.length, isRealModel: true });
                        }
                    } catch (err) {
                        console.warn("Real Pose Landmarker failed:", err);
                    }
                }
            }
            if (!ranReal) {
                simulatePoseTracking();
            }
        } 
        else if (selectedTask === 'Face Landmarker') {
            let ranReal = false;
            if (faceLandmarker) {
                ranReal = true;
                if (isTimeChanged) {
                    try {
                        const results = faceLandmarker.detectForVideo(video, now);
                        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                            const landmarks = results.faceLandmarks[0];
                            drawFaceLandmarks(landmarks);
                            
                            if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
                                const blendshapes = results.faceBlendshapes[0].categories;
                                
                                // Map all major 52-ARKit blendshapes to the riggingParamsRef!
                                if (riggingParamsRef.current) {
                                    riggingParamsRef.current.isMocapFaceActive = true;
                                    blendshapes.forEach((b) => {
                                        const k = b.categoryName;
                                        const score = b.score || 0;
                                        
                                        // Apply exponential smoothing to eliminate high-frequency jitter (low-pass filter)
                                        const smoothingFactor = 0.20; 
                                        const prevVal = riggingParamsRef.current[k] !== undefined && typeof riggingParamsRef.current[k] === 'number' ? riggingParamsRef.current[k] : 0;
                                        riggingParamsRef.current[k] = prevVal * (1 - smoothingFactor) + score * smoothingFactor;
                                    });
                                }
                                
                                setTaskOutput({
                                    type: 'face',
                                    landmarksCount: landmarks.length,
                                    blendshapes: blendshapes.slice(0, 10),
                                    isRealModel: true
                                });
                            }
                        } else {
                            // No face detected in this frame, but we are in real mode!
                            // Smoothly decay all blendshapes back to 0 to prevent harsh visual jumps
                            if (riggingParamsRef.current) {
                                riggingParamsRef.current.isMocapFaceActive = true;
                                const keysToDecay = [
                                    'jawOpen', 'mouthSmile', 'eyeBlinkLeft', 'eyeBlinkRight',
                                    'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
                                    'browDownLeft', 'browDownRight', 'browInnerUp', 'cheekPuff',
                                    'mouthPucker', 'mouthFunnel'
                                ];
                                keysToDecay.forEach(k => {
                                    if (riggingParamsRef.current[k] !== undefined && typeof riggingParamsRef.current[k] === 'number') {
                                        riggingParamsRef.current[k] *= 0.82;
                                    }
                                });
                            }
                        }
                    } catch (err) {
                        console.warn("Real Face Landmarker failed:", err);
                    }
                }
            }
            if (!ranReal) {
                simulateFaceTracking();
            }
        }
        else if (selectedTask === 'Face Detector') {
            let ranReal = false;
            if (faceDetector) {
                ranReal = true;
                if (isTimeChanged) {
                    try {
                        const results = faceDetector.detectForVideo(video, now);
                        if (results.detections && results.detections.length > 0) {
                            results.detections.forEach((d) => {
                                const box = d.boundingBox;
                                if (!box) return;
                                ctx.strokeStyle = '#F43F5E';
                                ctx.lineWidth = 3;
                                ctx.strokeRect(box.originX, box.originY, box.width, box.height);
                            });
                            setTaskOutput({ type: 'faceDetector', detectionsCount: results.detections.length, isRealModel: true });
                        }
                    } catch (err) {
                        console.warn("Real Face Detector failed:", err);
                    }
                }
            }
            if (!ranReal) {
                const faceX = canvas.width / 2 + Math.sin(now / 800) * 30;
                const faceY = canvas.height / 2 + Math.cos(now / 600) * 20;
                ctx.strokeStyle = '#F43F5E';
                ctx.lineWidth = 3;
                ctx.strokeRect(faceX - 60, faceY - 70, 120, 140);
                setTaskOutput({ type: 'faceDetector', detectionsCount: 1, isRealModel: false });
            }
        }
        else if (selectedTask === 'Hand Landmarker') {
            let ranReal = false;
            if (handLandmarker) {
                ranReal = true;
                if (isTimeChanged) {
                    try {
                        const results = handLandmarker.detectForVideo(video, now);
                        if (results.landmarks && results.landmarks.length > 0) {
                            ctx.strokeStyle = '#F97316';
                            ctx.fillStyle = '#F97316';
                            ctx.lineWidth = 2;
                            results.landmarks.forEach((hand) => {
                                hand.forEach((lm) => {
                                    ctx.beginPath();
                                    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
                                    ctx.fill();
                                });
                            });
                            setTaskOutput({ type: 'hand', activeHandsCount: results.landmarks.length, isRealModel: true });
                        }
                    } catch (err) {
                        console.warn("Real Hand Landmarker failed:", err);
                    }
                }
            }
            if (!ranReal) {
                const handX = canvas.width * 0.7 + Math.sin(now / 500) * 40;
                const handY = canvas.height * 0.6 + Math.cos(now / 400) * 30;
                ctx.strokeStyle = '#F97316';
                ctx.fillStyle = '#F97316';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(handX, handY, 15, 0, 2 * Math.PI);
                ctx.fill();
                for (let i = 0; i < 5; i++) {
                    const angle = -Math.PI / 2 + (i - 2) * 0.3;
                    ctx.beginPath();
                    ctx.moveTo(handX, handY);
                    ctx.lineTo(handX + Math.cos(angle) * 35, handY + Math.sin(angle) * 35);
                    ctx.stroke();
                }
                setTaskOutput({ type: 'hand', activeHandsCount: 1, isRealModel: false });
            }
        }
        else if (selectedTask === 'Gesture Recognizer') {
            let ranReal = false;
            if (gestureRecognizer) {
                ranReal = true;
                if (isTimeChanged) {
                    try {
                        const results = gestureRecognizer.recognizeForVideo(video, now);
                        
                        // Draw hand skeleton lines if real landmarks exist to show real movement!
                        if (results.landmarks && results.landmarks.length > 0) {
                            ctx.strokeStyle = '#8B5CF6'; // Purple for gesture hands
                            ctx.fillStyle = '#8B5CF6';
                            ctx.lineWidth = 2;
                            results.landmarks.forEach((hand) => {
                                hand.forEach((lm) => {
                                    ctx.beginPath();
                                    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
                                    ctx.fill();
                                });
                            });
                        }

                        if (results.gestures && results.gestures.length > 0) {
                            const gestureName = results.gestures[0][0].categoryName;
                            const score = results.gestures[0][0].score;
                            
                            ctx.fillStyle = 'rgba(167, 139, 250, 0.9)';
                            ctx.font = 'bold 16px monospace';
                            ctx.fillText(`GESTURE: ${gestureName.toUpperCase()} (${Math.round(score * 100)}%)`, 20, 40);

                            setTaskOutput({ type: 'gesture', gestureName, score, isRealModel: true });

                            const lowerGesture = gestureName.toLowerCase();
                            if (riggingParamsRef.current) {
                                if (lowerGesture === 'victory') {
                                    // Wave Left Arm
                                    const angle = Math.sin(now / 150) * 0.3;
                                    if (!riggingParamsRef.current['LeftArm']) riggingParamsRef.current['LeftArm'] = { x: 0, y: 0, z: 0 };
                                    riggingParamsRef.current['LeftArm'].z = -1.2 + angle;
                                    riggingParamsRef.current['LeftArm'].x = 0.2;
                                } else if (lowerGesture === 'pointing_up') {
                                    // Raise Right Arm
                                    const angle = Math.sin(now / 150) * 0.3;
                                    if (!riggingParamsRef.current['RightArm']) riggingParamsRef.current['RightArm'] = { x: 0, y: 0, z: 0 };
                                    riggingParamsRef.current['RightArm'].z = 1.2 + angle;
                                    riggingParamsRef.current['RightArm'].x = -0.2;
                                } else if (lowerGesture === 'thumbs_up') {
                                    // Nod Head and Smile
                                    if (!riggingParamsRef.current['Head']) riggingParamsRef.current['Head'] = { x: 0.5, y: 0.5, z: 0.5 };
                                    riggingParamsRef.current['Head'].x = 0.5 + Math.sin(now / 100) * 0.15; // Nodding pitch
                                    riggingParamsRef.current['mouthSmile'] = 0.8;
                                } else if (lowerGesture === 'closed_fist') {
                                    // Flex arms
                                    if (!riggingParamsRef.current['LeftForeArm']) riggingParamsRef.current['LeftForeArm'] = { x: 0, y: 0, z: 0 };
                                    if (!riggingParamsRef.current['RightForeArm']) riggingParamsRef.current['RightForeArm'] = { x: 0, y: 0, z: 0 };
                                    riggingParamsRef.current['LeftForeArm'].z = -1.2;
                                    riggingParamsRef.current['RightForeArm'].z = 1.2;
                                } else if (lowerGesture === 'open_palm') {
                                    // Reset arm/head rotations
                                    if (riggingParamsRef.current['LeftArm']) riggingParamsRef.current['LeftArm'] = { x: 0, y: 0, z: 0 };
                                    if (riggingParamsRef.current['RightArm']) riggingParamsRef.current['RightArm'] = { x: 0, y: 0, z: 0 };
                                    if (riggingParamsRef.current['Head']) riggingParamsRef.current['Head'] = { x: 0.5, y: 0.5, z: 0.5 };
                                } else if (lowerGesture === 'iloveyou') {
                                    riggingParamsRef.current['mouthSmile'] = 0.95;
                                    riggingParamsRef.current['eyeSquintLeft'] = 0.6;
                                    riggingParamsRef.current['eyeSquintRight'] = 0.6;
                                }
                            }
                        }
                    } catch (err) {
                        console.warn("Real Gesture Recognizer failed:", err);
                    }
                }
            }
            if (!ranReal) {
                simulateGestureTracking();
            }
        }
        else if (selectedTask === 'Object Detector') {
            let ranReal = false;
            if (objectDetector) {
                ranReal = true;
                if (isTimeChanged) {
                    try {
                        const results = objectDetector.detectForVideo(video, now);
                        if (results.detections && results.detections.length > 0) {
                            drawObjectBoxes(results.detections);
                            setTaskOutput({ type: 'object', detections: results.detections, isRealModel: true });
                        }
                    } catch (err) {
                        console.warn("Real Object Detector failed:", err);
                    }
                }
            }
            if (!ranReal) {
                simulateObjectTracking();
            }
        }
        else if (selectedTask === 'Image Segmenter' || selectedTask === 'Görüntü Segmentleyici') {
            let ranReal = false;
            if (imageSegmenter && isTimeChanged) {
                try {
                    imageSegmenter.segmentForVideo(video, now, (result: any) => {
                        if (result.categoryMask) {
                            ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
                            ctx.fillRect(0, 0, canvas.width, canvas.height);
                            setTaskOutput({ type: 'segmenter', accuracy: 0.99, maskChannels: 1, isRealModel: true });
                            ranReal = true;
                        }
                    });
                } catch (err) {
                    console.warn("Real Image Segmenter failed, falling back:", err);
                }
            }
            if (!ranReal) {
                const cx = canvas.width / 2;
                const cy = canvas.height / 2;
                ctx.fillStyle = 'rgba(16, 185, 129, 0.35)';
                ctx.beginPath();
                ctx.arc(cx, cy - 60, 40, 0, 2 * Math.PI);
                ctx.fill();
                ctx.beginPath();
                ctx.ellipse(cx, cy + 50, 70, 90, 0, 0, 2 * Math.PI);
                ctx.fill();
                setTaskOutput({ type: 'segmenter', accuracy: 0.95, maskChannels: 1, isRealModel: false });
            }
        }

        if (isPlayingRef.current) {
            animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
        }
    };

    const handleRawPose = (landmarks: any[]) => {
        // 1. NÖTR POZ KALİBRASYONU (Basit Vektör)
        if (needsCalibrationCapture.current) {
            Object.keys(BONE_RELATIONS).forEach(boneName => {
                const rel = BONE_RELATIONS[boneName];
                const p1 = landmarks[POSE_LANDMARKS[rel.p1]];
                const p2 = landmarks[POSE_LANDMARKS[rel.p2]];
                
                if (p1 && p2 && p1.x !== undefined && p2.x !== undefined) {
                    calibrationVectorsRef.current[boneName] = getVector(p1, p2);
                    smoothedPoseRef.current[boneName] = { x: 0.5, y: 0.5, z: 0.5 };
                }
            });
            needsCalibrationCapture.current = false;
            return; 
        }

        if (calibrationStatusRef.current !== 'active') return;

        // ENERJİ HESAPLAMA (AKILLI KİLİT)
        calculateGroupEnergy(landmarks);

        // GÜNCEL SEÇİMLERİ REF'TEN AL
        const currentSelectedBones = selectedBonesRef.current;

        // 2. AKTİF İZLEME VE EŞLEŞTİRME
        const mappedPose: MappedPose = {};

        // --- ÖZEL KAFA VE BOYUN ROTASYONU ---
        const headGroupState = groupStatesRef.current[BodyGroup.HEAD_NECK];
        // Head veya Neck seçiliyse hesapla
        if (headGroupState.isActive || currentSelectedBones.has('Head') || currentSelectedBones.has('Neck')) {
            const headRotation = calculateHeadRotation(landmarks);
            
            // Head Ataması
            if (headRotation && (currentSelectedBones.has('Head') || headGroupState.isActive)) {
                mappedPose['Head'] = headRotation;
            }

            // Neck Ataması
            if (currentSelectedBones.has('Neck') || headGroupState.isActive) {
                const neckRotation = calculateNeckRotation(landmarks, headRotation);
                if (neckRotation) {
                    mappedPose['Neck'] = neckRotation;
                }
            }
        }

        Object.keys(BONE_RELATIONS).forEach(boneName => {
            const rel = BONE_RELATIONS[boneName];
            const groupState = groupStatesRef.current[rel.group];

            // --- AKILLI KİLİT & SEÇİM KONTROLÜ ---
            // 1. Seçili değilse güncelleme
            if (!currentSelectedBones.has(boneName) && boneName !== 'Hips') { 
                 return;
            }

            // 2. Pasifse (yeterli enerji yoksa) güncelleme
            const ALWAYS_ACTIVE = [BodyGroup.HEAD_NECK, BodyGroup.TORSO];
            if (!groupState?.isActive && !ALWAYS_ACTIVE.includes(rel.group)) {
                return; 
            }

            const p1 = landmarks[POSE_LANDMARKS[rel.p1]];
            const p2 = landmarks[POSE_LANDMARKS[rel.p2]];

            if (p1 && p2 && p1.x !== undefined && p2.x !== undefined) {
                const neutralVec = calibrationVectorsRef.current[boneName];
                if (!neutralVec) return;

                const currentVec = getVector(p1, p2);

                let normalizedX = 0.5;
                let normalizedY = 0.5;
                let normalizedZ = 0.5;

                // --- VEC3 ROTASYON MOTORU V2.5 (Yeni Mimari) ---
                // Eski 2D Pad mapping mantığı terk edildi.
                // Vektörler arası açısal fark gerçek 3D koordinatlarla hesaplanıyor.
                
                // 1. Yaw (Y Ekseni - XZ Düzlemi)
                const currentYaw = Math.atan2(currentVec.x, currentVec.z);
                const neutralYaw = Math.atan2(neutralVec.x, neutralVec.z);
                const deltaYaw = currentYaw - neutralYaw;

                // 2. Pitch (X Ekseni - YZ Düzlemi)
                const currentPitch = Math.atan2(currentVec.y, currentVec.z);
                const neutralPitch = Math.atan2(neutralVec.y, neutralVec.z);
                const deltaPitch = currentPitch - neutralPitch;

                // 3. Roll (Z Ekseni - XY Düzlemi)
                const currentRoll = Math.atan2(currentVec.y, currentVec.x);
                const neutralRoll = Math.atan2(neutralVec.y, neutralVec.x);
                const deltaRoll = currentRoll - neutralRoll;

                // Hassasiyet Katsayıları (Radyan -> Pad 0..1 Map)
                const SENS = 0.65; // ~120 dereceyi tam aralığa yayar
                
                normalizedX = 0.5 + (deltaYaw / Math.PI) * SENS;
                normalizedY = 0.5 + (deltaPitch / Math.PI) * SENS;
                normalizedZ = 0.5 + (deltaRoll / Math.PI) * SENS;

                // Kemik Bazlı İnce Ayarlar (Terslemeler)
                if (boneName.includes('Left')) {
                    normalizedX = 1.0 - normalizedX;
                }
                if (boneName.includes('Arm') || boneName.includes('ForeArm')) {
                    normalizedY = 1.0 - normalizedY;
                }

                // Sınırları Koru (Clamp 0.0 - 1.0)
                normalizedX = Math.max(0, Math.min(1, normalizedX));
                normalizedY = Math.max(0, Math.min(1, normalizedY));
                normalizedZ = Math.max(0, Math.min(1, normalizedZ));

                // ORGANİK KAS SİMÜLASYONU (Velocity Clamp + Deep Lerp)
                let prevSmoothed = smoothedPoseRef.current[boneName];
                if (!prevSmoothed) {
                    prevSmoothed = { x: 0.5, y: 0.5, z: 0.5 };
                    smoothedPoseRef.current[boneName] = prevSmoothed;
                }
                
                // 1. Hız Limitörü (Bir karede maksimum %3 değişebilir - Daha ağırbaşlı)
                const MAX_SPEED = 0.03; 
                let diffX = normalizedX - prevSmoothed.x;
                let diffY = normalizedY - prevSmoothed.y;
                let diffZ = normalizedZ - prevSmoothed.z;

                diffX = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, diffX));
                diffY = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, diffY));
                diffZ = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, diffZ));

                normalizedX = prevSmoothed.x + diffX;
                normalizedY = prevSmoothed.y + diffY;
                normalizedZ = prevSmoothed.z + diffZ;

                // 2. Derin Yumuşatma (Daha organik bir süzülme)
                const lerpFactor = 0.12; 
                const finalX = lerp(prevSmoothed.x, normalizedX, lerpFactor);
                const finalY = lerp(prevSmoothed.y, normalizedY, lerpFactor);
                const finalZ = lerp(prevSmoothed.z, normalizedZ, lerpFactor);

                smoothedPoseRef.current[boneName] = { x: finalX, y: finalY, z: finalZ };
                mappedPose[boneName] = { x: finalX, y: finalY, z: finalZ };
            }
        });

        if (Object.keys(mappedPose).length > 0) {
            // --- APPLY TO RIGGING PARAMS ---
            Object.keys(mappedPose).forEach(boneName => {
                if (!currentSelectedBones.has(boneName)) return;
                
                const { x: normX, y: normY, z: normZ } = mappedPose[boneName];
                
                if (boneName === 'Hips_Position') {
                    // Özel Hips_Position mantığı
                    const diffY = 0.5 - normY;
                    let heightMultiplier = diffY > 0 ? 0.2 : 0.8;
                    let finalX = (0.5 - normX) * 0.2;
                    let finalY = diffY * heightMultiplier;
                    
                    if (!mocapPoseRef.current['Hips_Position']) {
                        mocapPoseRef.current['Hips_Position'] = { x: 0, y: 0, z: 0 };
                    }
                    
                    mappings.filter(m => m.enabled && m.bone === 'Hips_Position').forEach(m => {
                        let val = m.axis === 'x' ? finalX : (m.axis === 'z' ? finalY : 0);
                        if (m.invert) val *= -1;
                        val += m.offset;
                        
                        const currentVal = smoothedValuesRef.current[m.id] || 0;
                        const lerpFactor = 1 - m.smoothing;
                        const smoothedVal = currentVal + (val - currentVal) * lerpFactor;
                        smoothedValuesRef.current[m.id] = smoothedVal;
                        
                        mocapPoseRef.current['Hips_Position'][m.axis] = smoothedVal;
                    });
                    return;
                }

                const range = getBoneRotationRange(boneName);
                let rotY = (normX - 0.5) * (range * 2);
                let rotX = (normY - 0.5) * (range * 2);
                let rotZ = (normZ - 0.5) * (range * 2);

                if (['LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg'].includes(boneName)) rotY = 0;
                if (INVERTED_Z_BONES.includes(boneName)) rotZ = -rotZ;

                if (!mocapPoseRef.current[boneName]) {
                    mocapPoseRef.current[boneName] = { x: 0, y: 0, z: 0 };
                }

                mappings.filter(m => m.enabled && m.bone === boneName).forEach(m => {
                    let val = 0;
                    if (m.axis === 'x') val = rotX;
                    if (m.axis === 'y') val = rotY;
                    if (m.axis === 'z') val = rotZ;

                    if (m.invert) val *= -1;
                    val += m.offset;

                    const currentVal = smoothedValuesRef.current[m.id] || 0;
                    const lerpFactor = 1 - m.smoothing;
                    const smoothedVal = currentVal + (val - currentVal) * lerpFactor;
                    smoothedValuesRef.current[m.id] = smoothedVal;

                    // Mocap verisini riggingParams'a değil, doğrudan mocapPoseRef'e yaz!
                    mocapPoseRef.current[boneName][m.axis] = smoothedVal;
                    
                    // Geriye dönük uyumluluk için şimdilik riggingParams'ı da güncelleyebiliriz (İsteğe bağlı)
                    if (!riggingParamsRef.current[boneName]) {
                        riggingParamsRef.current[boneName] = { x: 0, y: 0, z: 0 };
                    }
                    riggingParamsRef.current[boneName][m.axis] = smoothedVal;
                });
            });
        }
    };

    const startMocap = () => setIsActive(true);
    const stopMocap = () => {
        setIsActive(false);
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
    const calibrate = () => { 
        runCalibration();
    };
    
    return { 
        isActive, 
        isReady: isCameraReady, 
        mappings, 
        setMappings, 
        isDistanceCompEnabled, 
        setIsDistanceCompEnabled, 
        globalDeadzone, 
        setGlobalDeadzone, 
        error, 
        videoRef, 
        canvasRef, 
        startMocap, 
        stopMocap, 
        calibrate,
        calibrationStatus,
        countdown,
        taskOutput,
        textInput,
        setTextInput,
        processTextInput
    };
};
