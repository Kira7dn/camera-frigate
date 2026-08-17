import { BirdseyeConfig } from "@/types/frigateConfig";
import MSEPlayer from "./MsePlayer";
import { cn } from "@/lib/utils";
import React from "react";
import { ImageShadowOverlay } from "../overlay/ImageShadowOverlay";

type LivePlayerProps = {
  className?: string;
  birdseyeConfig: BirdseyeConfig;
  pip?: boolean;
  containerRef: React.MutableRefObject<HTMLDivElement | null>;
  playerRef?: React.MutableRefObject<HTMLDivElement | null>;
  onClick?: () => void;
};

export default function BirdseyeLivePlayer({
  className,
  birdseyeConfig,
  pip,
  containerRef,
  playerRef,
  onClick,
}: LivePlayerProps) {
  const player = (
    <MSEPlayer
      className="size-full rounded-lg md:rounded-2xl"
      camera="birdseye"
      pip={pip}
    />
  );

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative flex w-full cursor-pointer justify-center",
        className,
      )}
      onClick={onClick}
    >
      <ImageShadowOverlay
        upperClassName="md:rounded-2xl"
        lowerClassName="md:rounded-2xl"
      />
      <div className="size-full" ref={playerRef}>
        {player}
      </div>
    </div>
  );
}
