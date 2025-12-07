'use client';

import { createContext, useCallback, useContext, useMemo, useState, ReactNode, useEffect } from 'react';

type NotificationVariant = 'success' | 'error' | 'warning' | 'info';

interface Notification {
  id: string;
  title?: string;
  message: string;
  variant: NotificationVariant;
  duration?: number;
}

interface NotificationContextValue {
  notify: (message: string, options?: { title?: string; variant?: NotificationVariant; duration?: number }) => void;
  dismiss: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextValue | undefined>(undefined);

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const dismiss = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((notification) => notification.id !== id));
  }, []);

  const notify = useCallback((message: string, options?: { title?: string; variant?: NotificationVariant; duration?: number }) => {
    const id =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const variant = options?.variant ?? 'info';
    const duration = options?.duration ?? 3500;

    const notification: Notification = {
      id,
      message,
      title: options?.title,
      variant,
      duration
    };

    setNotifications((prev) => [...prev, notification]);

    if (duration > 0) {
      setTimeout(() => dismiss(id), duration);
    }
  }, [dismiss]);

  const value = useMemo(() => ({ notify, dismiss }), [notify, dismiss]);

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <div className="fixed inset-x-0 top-4 z-[9999] flex flex-col items-center gap-3 px-4 sm:items-end sm:px-6">
        {notifications.map((notification) => (
          <NotificationToast key={notification.id} notification={notification} onDismiss={dismiss} />
        ))}
      </div>
    </NotificationContext.Provider>
  );
}

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

function NotificationToast({ notification, onDismiss }: { notification: Notification; onDismiss: (id: string) => void }) {
  const { id, title, message, variant } = notification;

  const variantStyles: Record<NotificationVariant, string> = {
    success: 'bg-green-50 text-green-900 border border-green-200',
    error: 'bg-red-50 text-red-900 border border-red-200',
    warning: 'bg-yellow-50 text-yellow-900 border border-yellow-200',
    info: 'bg-blue-50 text-blue-900 border border-blue-200'
  };

  const iconMap: Record<NotificationVariant, string> = {
    success: '✓',
    error: '!',
    warning: '⚠',
    info: 'ℹ'
  };

  return (
    <div
      className={`w-full max-w-sm rounded-xl px-4 py-3 shadow-lg transition-all animate-slide-in ${variantStyles[variant]}`}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <div className="flex h-6 w-6 items-center justify-center rounded-full bg-white/60 text-sm font-bold text-gray-700">
          {iconMap[variant]}
        </div>
        <div className="flex-1">
          {title && <p className="text-sm font-semibold">{title}</p>}
          <p className="text-sm leading-snug">{message}</p>
        </div>
        <button
          type="button"
          onClick={() => onDismiss(id)}
          className="text-sm text-gray-500 transition-colors hover:text-gray-800"
          aria-label="Dismiss notification"
        >
          ×
        </button>
      </div>
    </div>
  );
}

