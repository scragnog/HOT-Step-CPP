import React, { createContext, useContext, useState, type ReactNode } from 'react';

type Language = 'en' | 'vi';

interface Translations {
  [key: string]: {
    en: string;
    vi: string;
  };
}

export const translations: Translations = {
  // Sidebar
  nav_create: { en: 'Create', vi: 'Sáng tác' },
  nav_library: { en: 'Library', vi: 'Thư viện' },
  nav_lyric_studio: { en: 'Lyric Studio', vi: 'Studio Lời' },
  nav_settings: { en: 'Settings', vi: 'Cài đặt' },
  nav_terminal: { en: 'Terminal', vi: 'Terminal' },
  nav_quit: { en: 'Quit', vi: 'Thoát' },
  
  // Create Panel
  create_title: { en: 'Create New Track', vi: 'Tạo bài hát mới' },
  create_description_label: { en: 'Style Description', vi: 'Mô tả phong cách' },
  create_description_placeholder: { en: 'e.g. upbeat pop, 80s synthwave...', vi: 'ví dụ: nhạc pop sôi động, synthwave thập niên 80...' },
  create_lyrics_label: { en: 'Lyrics', vi: 'Lời bài hát' },
  create_lyrics_placeholder: { en: 'Paste your lyrics here...', vi: 'Dán lời bài hát vào đây...' },
  create_instrumental: { en: 'Instrumental', vi: 'Không lời' },
  create_generate_btn: { en: 'Generate', vi: 'Tạo nhạc' },
  create_generating: { en: 'Generating...', vi: 'Đang tạo...' },
  create_song_title_label: { en: 'Song Title', vi: 'Tên bài hát' },
  create_song_title_placeholder: { en: 'Untitled', vi: 'Không tên' },

  // Sections
  section_models: { en: 'Models', vi: 'Mẫu (Models)' },
  section_gen_settings: { en: 'Generation Settings', vi: 'Cài đặt tạo' },
  section_mastering: { en: 'Mastering', vi: 'Mastering' },
  section_adapters: { en: 'Adapters', vi: 'Adapters / LoRA' },

  // Metadata
  meta_bpm: { en: 'BPM', vi: 'BPM' },
  meta_key: { en: 'Key / Scale', vi: 'Tone / Scale' },
  meta_duration: { en: 'Duration (sec)', vi: 'Thời lượng (giây)' },
  meta_language: { en: 'Vocal Language', vi: 'Ngôn ngữ hát' },
  meta_time_sig: { en: 'Time Signature', vi: 'Nhịp (Time Sig)' },
  meta_seed: { en: 'Seed', vi: 'Seed' },
  meta_random: { en: 'Random', vi: 'Ngẫu nhiên' },

  // Common
  btn_cancel: { en: 'Cancel', vi: 'Hủy' },
  btn_close: { en: 'Close', vi: 'Đóng' },
  btn_delete: { en: 'Delete', vi: 'Xóa' },
  btn_rename: { en: 'Rename', vi: 'Đổi tên' },
  btn_save: { en: 'Save', vi: 'Lưu' },
  
  // Notifications
  notif_deleted: { en: 'Deleted', vi: 'Đã xóa' },
  notif_complete: { en: 'Complete!', vi: 'Hoàn tất!' },
  
  // Settings
  settings_title: { en: 'Settings', vi: 'Cài đặt hệ thống' },
  settings_general: { en: 'General', vi: 'Chung' },
  settings_audio: { en: 'Audio', vi: 'Âm thanh' },
  settings_appearance: { en: 'Appearance', vi: 'Giao diện' },
};

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<Language>(() => {
    return (localStorage.getItem('hs-language') as Language) || 'en';
  });

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem('hs-language', lang);
  };

  const t = (key: string): string => {
    if (!translations[key]) return key;
    return translations[key][language];
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) throw new Error('useLanguage must be used within LanguageProvider');
  return context;
};
