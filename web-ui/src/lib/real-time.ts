import { useState, useEffect, useCallback, useRef } from 'react';

// Types for real-time events
export interface RealTimeEvent {
    type: string;
    data: any;
    timestamp: number;
}

export interface UseRealTimeOptions {
    enabled?: boolean;
    retryAttempts?: number;
    retryDelay?: number;
    onEvent?: (event: RealTimeEvent) => void;
    onError?: (error: Event) => void;
    onOpen?: () => void;
    onClose?: () => void;
}

// Hook for Server-Sent Events (SSE)
export function useRealTime(url: string | null, options: UseRealTimeOptions = {}) {
    const {
        enabled = true,
        retryAttempts = 3,
        retryDelay = 1000,
        onEvent,
        onError,
        onOpen,
        onClose,
    } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [lastEvent, setLastEvent] = useState<RealTimeEvent | null>(null);
    const [error, setError] = useState<Event | null>(null);

    const eventSourceRef = useRef<EventSource | null>(null);
    const retryCountRef = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        if (!url || !enabled || eventSourceRef.current) return;

        try {
            const eventSource = new EventSource(url);
            eventSourceRef.current = eventSource;

            eventSource.onopen = () => {
                setIsConnected(true);
                setError(null);
                retryCountRef.current = 0;
                onOpen?.();
            };

            eventSource.onmessage = (event) => {
                try {
                    const parsedData = JSON.parse(event.data);
                    const realTimeEvent: RealTimeEvent = {
                        type: event.type || 'message',
                        data: parsedData,
                        timestamp: Date.now(),
                    };

                    setLastEvent(realTimeEvent);
                    onEvent?.(realTimeEvent);
                } catch (error) {
                    console.error('Error parsing SSE message:', error);
                }
            };

            eventSource.onerror = (error) => {
                setIsConnected(false);
                setError(error);
                onError?.(error);

                // Auto-reconnect logic
                if (retryCountRef.current < retryAttempts) {
                    retryCountRef.current++;
                    reconnectTimeoutRef.current = setTimeout(() => {
                        eventSource.close();
                        eventSourceRef.current = null;
                        connect();
                    }, retryDelay * retryCountRef.current);
                } else {
                    eventSource.close();
                    eventSourceRef.current = null;
                }
            };

            // Note: EventSource doesn't have onclose, so we handle closure in onerror
        } catch (error) {
            console.error('Error creating EventSource:', error);
            setError(error as Event);
        }
    }, [url, enabled, retryAttempts, retryDelay, onEvent, onError, onOpen, onClose]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }

        setIsConnected(false);
        onClose?.();
    }, [onClose]);

    const reconnect = useCallback(() => {
        disconnect();
        retryCountRef.current = 0;
        connect();
    }, [disconnect, connect]);

    useEffect(() => {
        if (enabled && url) {
            connect();
        } else {
            disconnect();
        }

        return disconnect;
    }, [enabled, url, connect, disconnect]);

    return {
        isConnected,
        lastEvent,
        error,
        reconnect,
        disconnect,
    };
}

// Hook for real-time dashboard updates
export function useRealTimeDashboard() {
    return useRealTime('/api/dashboard/realtime', {
        onEvent: (event) => {
            if (event.type === 'dashboard_update') {
                // Handle dashboard update
                console.log('Dashboard updated:', event.data);
            }
        },
    });
}

// Hook for real-time pool updates
export function useRealTimePools() {
    return useRealTime('/api/pools/realtime', {
        onEvent: (event) => {
            if (event.type === 'pool_update') {
                // Handle pool update
                console.log('Pool updated:', event.data);
            } else if (event.type === 'pool_created') {
                // Handle new pool
                console.log('New pool:', event.data);
            } else if (event.type === 'pool_deleted') {
                // Handle pool deletion
                console.log('Pool deleted:', event.data);
            }
        },
    });
}

// Hook for real-time optimization run updates
export function useRealTimeOptimizationRuns() {
    return useRealTime('/api/optimization/runs/realtime', {
        onEvent: (event) => {
            if (event.type === 'run_update') {
                // Handle optimization run update
                console.log('Optimization run updated:', event.data);
            } else if (event.type === 'run_created') {
                // Handle new optimization run
                console.log('New optimization run:', event.data);
            }
        },
    });
}

// WebSocket fallback for browsers that don't support SSE
export function useWebSocket(url: string | null, options: UseRealTimeOptions = {}) {
    const {
        enabled = true,
        retryAttempts = 3,
        retryDelay = 1000,
        onEvent,
        onError,
        onOpen,
        onClose,
    } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [lastEvent, setLastEvent] = useState<RealTimeEvent | null>(null);
    const [error, setError] = useState<Event | null>(null);

    const webSocketRef = useRef<WebSocket | null>(null);
    const retryCountRef = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        if (!url || !enabled || webSocketRef.current) return;

        try {
            const ws = new WebSocket(url);
            webSocketRef.current = ws;

            ws.onopen = () => {
                setIsConnected(true);
                setError(null);
                retryCountRef.current = 0;
                onOpen?.();
            };

            ws.onmessage = (event) => {
                try {
                    const parsedData = JSON.parse(event.data);
                    const realTimeEvent: RealTimeEvent = {
                        type: parsedData.type || 'message',
                        data: parsedData.data || parsedData,
                        timestamp: Date.now(),
                    };

                    setLastEvent(realTimeEvent);
                    onEvent?.(realTimeEvent);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            ws.onerror = (error) => {
                setIsConnected(false);
                setError(error as Event);
                onError?.(error);

                // Auto-reconnect logic
                if (retryCountRef.current < retryAttempts) {
                    retryCountRef.current++;
                    reconnectTimeoutRef.current = setTimeout(() => {
                        ws.close();
                        webSocketRef.current = null;
                        connect();
                    }, retryDelay * retryCountRef.current);
                } else {
                    ws.close();
                    webSocketRef.current = null;
                }
            };

            ws.onclose = () => {
                setIsConnected(false);
                onClose?.();
            };
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            setError(error as Event);
        }
    }, [url, enabled, retryAttempts, retryDelay, onEvent, onError, onOpen, onClose]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (webSocketRef.current) {
            webSocketRef.current.close();
            webSocketRef.current = null;
        }

        setIsConnected(false);
    }, []);

    const reconnect = useCallback(() => {
        disconnect();
        retryCountRef.current = 0;
        connect();
    }, [disconnect, connect]);

    const send = useCallback((data: any) => {
        if (webSocketRef.current && isConnected) {
            webSocketRef.current.send(JSON.stringify(data));
        }
    }, [isConnected]);

    useEffect(() => {
        if (enabled && url) {
            connect();
        } else {
            disconnect();
        }

        return disconnect;
    }, [enabled, url, connect, disconnect]);

    return {
        isConnected,
        lastEvent,
        error,
        reconnect,
        disconnect,
        send,
    };
}

// Utility function to check SSE support
export function supportsSSE(): boolean {
    return typeof EventSource !== 'undefined';
}

// Utility function to check WebSocket support
export function supportsWebSocket(): boolean {
    return typeof WebSocket !== 'undefined';
}

// Hook that chooses between SSE and WebSocket based on browser support
export function useRealTimeAuto(url: string | null, options: UseRealTimeOptions = {}) {
    const sseHook = useRealTime(supportsSSE() ? url : null, options);
    const wsHook = useWebSocket(supportsSSE() ? null : url, options);

    return supportsSSE() ? sseHook : wsHook;
}

// Real-time event types
export const REAL_TIME_EVENTS = {
    DASHBOARD_UPDATE: 'dashboard_update',
    POOL_UPDATE: 'pool_update',
    POOL_CREATED: 'pool_created',
    POOL_DELETED: 'pool_deleted',
    RUN_UPDATE: 'run_update',
    RUN_CREATED: 'run_created',
    TOKEN_UPDATE: 'token_update',
    PROTOCOL_UPDATE: 'protocol_update',
    CONFIG_UPDATE: 'config_update',
} as const;