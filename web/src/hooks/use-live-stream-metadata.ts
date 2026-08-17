import { CameraConfig, FrigateConfig } from "@/types/frigateConfig";
import { useMemo } from "react";
import useSWR from "swr";
import useDeferredStreamMetadata from "./use-deferred-stream-metadata";
import { detectCameraAudioFeatures } from "@/utils/cameraUtil";

export default function useLiveStreamMetadata(
  cameras: CameraConfig[],
  activeStreams?: { [cameraName: string]: string },
) {
  const { data: config } = useSWR<FrigateConfig>("config");

  const streamNames = useMemo(() => {
    if (!config) return [];

    const names = new Set<string>();
    cameras.forEach((camera) => {
      const selected = activeStreams?.[camera.name];
      const streams = selected ? [selected] : Object.values(camera.live.streams);
      streams.forEach((streamName) => {
        if (
          Object.prototype.hasOwnProperty.call(
            config.go2rtc.streams || {},
            streamName,
          )
        ) {
          names.add(streamName);
        }
      });
    });
    return Array.from(names);
  }, [activeStreams, cameras, config]);

  const streamMetadata = useDeferredStreamMetadata(streamNames);

  const isRestreamedStates = useMemo(() => {
    const states: Record<string, boolean> = {};
    cameras.forEach((camera) => {
      const selected =
        activeStreams?.[camera.name] ?? Object.values(camera.live.streams)[0];
      states[camera.name] = Boolean(
        config &&
          Object.prototype.hasOwnProperty.call(
            config.go2rtc.streams || {},
            selected,
          ),
      );
    });
    return states;
  }, [activeStreams, cameras, config]);

  const supportsAudioOutputStates = useMemo(() => {
    const states: Record<
      string,
      { supportsAudio: boolean; cameraName: string }
    > = {};
    cameras.forEach((camera) => {
      Object.values(camera.live.streams).forEach((streamName) => {
        states[streamName] = {
          supportsAudio: detectCameraAudioFeatures(streamMetadata[streamName])
            .audioOutput,
          cameraName: camera.name,
        };
      });
    });
    return states;
  }, [cameras, streamMetadata]);

  return { isRestreamedStates, supportsAudioOutputStates, streamMetadata };
}
