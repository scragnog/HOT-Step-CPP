// CoverImage.tsx — a song's artwork, with the generic stand-in already applied.
//
// Every place that shows cover art uses this, so "what do we draw when there is
// no cover" is decided once. If the image itself fails to load the note icon
// comes back, which is what dev checkouts without the covers folder will see.

import React, { useEffect, useState } from 'react';
import { Music } from 'lucide-react';
import { coverSrc, useCoverSet } from '../../utils/defaultCover';

interface CoverImageProps {
  /** The song's own cover, if it has one. */
  url?: string | null;
  /** Stable per-song value — the id. Decides which generic cover it gets. */
  seed: string;
  className?: string;
  alt?: string;
  /** Size of the fallback note icon, in px. */
  iconSize?: number;
}

export const CoverImage: React.FC<CoverImageProps> = ({
  url, seed, className = 'w-full h-full object-cover', alt = '', iconSize = 18,
}) => {
  const [broken, setBroken] = useState(false);
  const set = useCoverSet();
  const src = coverSrc(url, seed, set);

  // A new src deserves a fresh attempt — switching sets should not inherit the
  // previous one's failure.
  useEffect(() => { setBroken(false); }, [src]);

  if (broken) return <Music size={iconSize} className="text-zinc-600" />;

  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading="lazy"
      onError={() => setBroken(true)}
    />
  );
};
