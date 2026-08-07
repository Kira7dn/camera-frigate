import axios from "axios";
import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { toast } from "sonner";

import { SettingsGroupCard } from "@/components/card/SettingsGroupCard";
import FilterSwitch from "@/components/filter/FilterSwitch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type {
  FrigateConfig,
  NotificationEvent,
  NotificationRecipientConfig,
  NotificationRuleConfig,
} from "@/types/frigateConfig";
import { useNotificationSuspend } from "@/api/ws";
import type { SectionRendererProps } from "./registry";

type ChannelName = "webpush" | "telegram" | "zalo";
type NotificationDocument = FrigateConfig["notifications"];

type ProviderStatus = {
  enabled: boolean;
  configured: boolean;
  readiness: "ready" | "missing" | "degraded";
  pending: number;
  last_success?: string | null;
  last_error?: string | null;
};

type ConfigResponse = {
  revision: string;
  notifications: NotificationDocument;
  capabilities: { cameras: string[]; events: NotificationEvent[] };
  providers: Record<ChannelName, ProviderStatus>;
};

const EVENTS: { value: NotificationEvent; label: string }[] = [
  { value: "alert", label: "Alert" },
  { value: "object_detected", label: "Object detected" },
  { value: "license_plate", label: "License plate" },
  { value: "face_recognized", label: "Face recognized" },
  { value: "camera_offline", label: "Camera offline" },
  { value: "camera_online", label: "Camera online" },
  { value: "semantic_trigger", label: "Semantic trigger" },
  { value: "camera_monitoring", label: "Camera monitoring" },
];
const NOTIFICATION_SERVICE_WORKER = "/notifications-worker.js";

const clone = <T,>(value: T): T => structuredClone(value);

function newRule(index: number): NotificationRuleConfig {
  return {
    id: `rule_${index}`,
    name: `Notification rule ${index}`,
    enabled: true,
    event: "alert",
    filters: {
      cameras: [],
      labels: [],
      zones: [],
      identities: [],
      trigger_names: [],
      conditions: [],
    },
    destinations: { webpush: true, telegram: [], zalo: [] },
    cooldown: 30,
  };
}

export default function NotificationsSettingsExtras({
  selectedCamera,
  formContext,
}: SectionRendererProps) {
  const { data, mutate } = useSWR<ConfigResponse>("notifications/config", {
    revalidateOnFocus: false,
  });
  const [document, setDocument] = useState<NotificationDocument>();
  const [revision, setRevision] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!data || dirty) return;
    setDocument(clone(data.notifications));
    setRevision(data.revision);
  }, [data, dirty]);

  const update = (mutator: (draft: NotificationDocument) => void) => {
    if (!document) return;
    const draft = clone(document);
    mutator(draft);
    setDocument(draft);
    setDirty(true);
  };

  const save = async () => {
    if (!document) return;
    setSaving(true);
    try {
      const response = await axios.put("notifications/config", {
        revision,
        notifications: document,
      });
      setDocument(clone(response.data.notifications));
      setRevision(response.data.revision);
      setDirty(false);
      await mutate();
      toast.success("Notification configuration saved to deploy/config.yaml");
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 409) {
        toast.error("config.yaml changed. Reload before saving again.");
      } else {
        toast.error("Unable to save notification configuration");
      }
    } finally {
      setSaving(false);
    }
  };

  if (!document || !data) {
    return <div className="p-4 text-sm text-primary-variant">Loading notifications…</div>;
  }

  if (formContext?.level === "camera" && selectedCamera) {
    return (
      <CameraNotificationView
        camera={selectedCamera}
        document={document}
      />
    );
  }

  return (
    <div className="space-y-6 pb-8">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-secondary-variant bg-secondary p-4">
        <div>
          <div className="font-medium">Notification configuration</div>
          <div className="text-sm text-primary-variant">
            Single source of truth: deploy/config.yaml
          </div>
        </div>
        <div className="flex gap-2">
          {dirty && (
            <Button
              variant="outline"
              onClick={() => {
                setDocument(clone(data.notifications));
                setRevision(data.revision);
                setDirty(false);
              }}
            >
              Discard
            </Button>
          )}
          <Button disabled={!dirty || saving} onClick={save}>
            {saving ? "Saving…" : "Save Notifications"}
          </Button>
        </div>
      </div>

      <SettingsGroupCard title="General">
        <div className="space-y-3">
          <FilterSwitch
            label="Enable notifications"
            isChecked={document.enabled}
            onCheckedChange={(enabled) =>
              update((draft) => (draft.enabled = enabled))
            }
          />
          <label className="block max-w-xl space-y-1 text-sm">
            <span>WebPush contact email</span>
            <Input
              type="email"
              value={document.email ?? ""}
              placeholder="admin@example.com"
              onChange={(event) =>
                update((draft) => (draft.email = event.target.value || null))
              }
            />
          </label>
        </div>
      </SettingsGroupCard>

      <SettingsGroupCard title="Channels">
        <div className="grid gap-4 xl:grid-cols-3">
          {(["webpush", "telegram", "zalo"] as ChannelName[]).map((channel) => (
            <ChannelCard
              key={channel}
              channel={channel}
              document={document}
              status={data.providers[channel]}
              dirty={dirty}
              update={update}
              refresh={() => mutate()}
            />
          ))}
        </div>
      </SettingsGroupCard>

      <SettingsGroupCard title="Rules">
        <div className="space-y-4">
          {document.rules.map((rule, index) => (
            <RuleEditor
              key={`${rule.id}-${index}`}
              rule={rule}
              cameras={data.capabilities.cameras}
              channels={document.channels}
              dirty={dirty}
              updateRule={(next) =>
                update((draft) => {
                  draft.rules[index] = next;
                })
              }
              remove={() => update((draft) => draft.rules.splice(index, 1))}
              refresh={() => mutate()}
            />
          ))}
          <Button
            variant="outline"
            onClick={() => update((draft) => draft.rules.push(newRule(draft.rules.length + 1)))}
          >
            Add rule
          </Button>
        </div>
      </SettingsGroupCard>

      <SettingsGroupCard title="Delivery">
        <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-5">
          {(
            [
              ["max_attempts", "Maximum attempts"],
              ["initial_backoff", "Initial backoff (s)"],
              ["max_backoff", "Maximum backoff (s)"],
              ["retention_days", "Retention (days)"],
              ["max_pending", "Maximum pending"],
            ] as const
          ).map(([field, label]) => (
            <label key={field} className="space-y-1 text-sm">
              <span>{label}</span>
              <Input
                type="number"
                min={1}
                value={document.delivery[field]}
                onChange={(event) =>
                  update(
                    (draft) =>
                      (draft.delivery[field] = Number(event.target.value)),
                  )
                }
              />
            </label>
          ))}
        </div>
      </SettingsGroupCard>
    </div>
  );
}

function CameraNotificationView({
  camera,
  document,
}: {
  camera: string;
  document: NotificationDocument;
}) {
  const { payload: suspendedUntil, send: sendSuspend } =
    useNotificationSuspend(camera);
  const rules = document.rules.filter(
    (rule) =>
      rule.enabled &&
      (rule.filters.cameras.length === 0 ||
        rule.filters.cameras.includes(camera)),
  );
  const suspended = Boolean(suspendedUntil && suspendedUntil !== "0");
  return (
    <div className="space-y-4 pb-8">
      <SettingsGroupCard title={`Notifications for ${camera}`}>
        <div className="space-y-3">
          <div className="text-sm text-primary-variant">
            Persistent settings are managed by rules in Settings → Notifications.
          </div>
          {rules.length ? (
            rules.map((rule) => (
              <div
                key={rule.id}
                className="flex items-center justify-between rounded-md border border-secondary-variant p-3"
              >
                <div>
                  <div className="font-medium">{rule.name}</div>
                  <div className="text-xs text-primary-variant">
                    {rule.event} · {rule.cooldown}s cooldown
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-sm">No notification rule applies to this camera.</div>
          )}
          <div className="flex flex-wrap items-center gap-2 pt-2">
            <div className={suspended ? "text-danger" : "text-success"}>
              {suspended ? `Suspended until ${suspendedUntil}` : "Active"}
            </div>
            {!suspended ? (
              <>
                <Button size="sm" variant="outline" onClick={() => sendSuspend(5)}>
                  Suspend 5 minutes
                </Button>
                <Button size="sm" variant="outline" onClick={() => sendSuspend(60)}>
                  Suspend 1 hour
                </Button>
              </>
            ) : (
              <Button size="sm" variant="outline" onClick={() => sendSuspend(0)}>
                Resume
              </Button>
            )}
          </div>
        </div>
      </SettingsGroupCard>
    </div>
  );
}

function ChannelCard({
  channel,
  document,
  status,
  dirty,
  update,
  refresh,
}: {
  channel: ChannelName;
  document: NotificationDocument;
  status?: ProviderStatus;
  dirty: boolean;
  update: (mutator: (draft: NotificationDocument) => void) => void;
  refresh: () => void;
}) {
  const channelConfig = document.channels[channel];
  const recipients = "recipients" in channelConfig ? channelConfig.recipients : [];
  const test = async (recipientId = "") => {
    try {
      await axios.post(`notifications/providers/${channel}/test`, {
        recipient_id: recipientId,
      });
      toast.success("Test notification queued");
      refresh();
    } catch {
      toast.error("Provider is not ready");
    }
  };

  return (
    <div className="space-y-3 rounded-lg border border-secondary-variant p-4">
      <div className="flex items-center justify-between">
        <div className="font-medium capitalize">{channel}</div>
        <div className="text-xs uppercase text-primary-variant">
          {status?.readiness ?? "loading"} · {status?.pending ?? 0} pending
        </div>
      </div>
      <FilterSwitch
        label="Enabled"
        isChecked={channelConfig.enabled}
        onCheckedChange={(enabled) =>
          update((draft) => (draft.channels[channel].enabled = enabled))
        }
      />
      {channel === "zalo" && (
        <label className="block space-y-1 text-sm">
          <span>Public base URL</span>
          <Input
            value={document.channels.zalo.public_base_url ?? ""}
            placeholder="https://camera.example.com"
            onChange={(event) =>
              update(
                (draft) =>
                  (draft.channels.zalo.public_base_url = event.target.value || null),
              )
            }
          />
        </label>
      )}
      {channel !== "webpush" && (
        <div className="space-y-3">
          {recipients.map((recipient, index) => (
            <RecipientEditor
              key={`${recipient.id}-${index}`}
              recipient={recipient}
              onChange={(next) =>
                update((draft) => {
                  const config = draft.channels[channel];
                  if ("recipients" in config) config.recipients[index] = next;
                })
              }
              remove={() =>
                update((draft) => {
                  const config = draft.channels[channel];
                  if ("recipients" in config) config.recipients.splice(index, 1);
                })
              }
              test={() => test(recipient.id)}
              testDisabled={dirty || !recipient.enabled || !status?.configured}
            />
          ))}
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              update((draft) => {
                const config = draft.channels[channel];
                if ("recipients" in config)
                  config.recipients.push({
                    id: `recipient_${config.recipients.length + 1}`,
                    name: "New recipient",
                    chat_id: "",
                    enabled: true,
                  });
              })
            }
          >
            Add recipient
          </Button>
        </div>
      )}
      {channel === "webpush" && (
        <div className="space-y-2">
          <WebPushDeviceRegistration enabled={channelConfig.enabled} />
          <Button
            size="sm"
            variant="outline"
            disabled={dirty || !status?.configured}
            onClick={() => test()}
          >
            Test saved configuration
          </Button>
        </div>
      )}
      {status?.last_error && (
        <div className="text-xs text-danger">{status.last_error}</div>
      )}
    </div>
  );
}

function WebPushDeviceRegistration({ enabled }: { enabled: boolean }) {
  const { data: publicKey } = useSWR<string>(
    enabled ? "notifications/pubkey" : null,
    { revalidateOnFocus: false },
  );
  const [registration, setRegistration] =
    useState<ServiceWorkerRegistration | null>();

  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    navigator.serviceWorker
      .getRegistration(NOTIFICATION_SERVICE_WORKER)
      .then((value) => setRegistration(value ?? null))
      .catch(() => setRegistration(null));
  }, []);

  const toggle = async () => {
    if (registration) {
      const subscription = await registration.pushManager.getSubscription();
      await subscription?.unsubscribe();
      await registration.unregister();
      setRegistration(null);
      toast.success("WebPush device unregistered");
      return;
    }
    if (!publicKey || !("Notification" in window) || !window.isSecureContext) {
      toast.error("WebPush requires HTTPS and a ready VAPID key");
      return;
    }
    const permission = await Notification.requestPermission();
    if (permission !== "granted") return;
    const worker = await navigator.serviceWorker.register(
      NOTIFICATION_SERVICE_WORKER,
      { updateViaCache: "none" },
    );
    await navigator.serviceWorker.ready;
    const subscription = await worker.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: publicKey,
    });
    await axios.post("notifications/register", { sub: subscription });
    setRegistration(worker);
    toast.success("WebPush device registered");
  };

  return (
    <Button
      size="sm"
      variant="outline"
      disabled={!enabled || !("serviceWorker" in navigator)}
      onClick={toggle}
    >
      {registration ? "Unregister this device" : "Register this device"}
    </Button>
  );
}

function RecipientEditor({
  recipient,
  onChange,
  remove,
  test,
  testDisabled,
}: {
  recipient: NotificationRecipientConfig;
  onChange: (recipient: NotificationRecipientConfig) => void;
  remove: () => void;
  test: () => void;
  testDisabled: boolean;
}) {
  return (
    <div className="space-y-2 rounded-md bg-secondary p-3">
      <div className="grid gap-2 md:grid-cols-2">
        <Input value={recipient.id} onChange={(e) => onChange({ ...recipient, id: e.target.value })} placeholder="ID" />
        <Input value={recipient.name} onChange={(e) => onChange({ ...recipient, name: e.target.value })} placeholder="Name" />
      </div>
      <Input value={recipient.chat_id} onChange={(e) => onChange({ ...recipient, chat_id: e.target.value })} placeholder="Chat ID or {ENV_NAME}" />
      <div className="flex flex-wrap items-center gap-2">
        <FilterSwitch label="Enabled" isChecked={recipient.enabled} onCheckedChange={(enabled) => onChange({ ...recipient, enabled })} />
        <Button size="sm" variant="outline" disabled={testDisabled} onClick={test}>Test</Button>
        <Button size="sm" variant="destructive" onClick={remove}>Remove</Button>
      </div>
    </div>
  );
}

function RuleEditor({
  rule,
  cameras,
  channels,
  dirty,
  updateRule,
  remove,
  refresh,
}: {
  rule: NotificationRuleConfig;
  cameras: string[];
  channels: NotificationDocument["channels"];
  dirty: boolean;
  updateRule: (rule: NotificationRuleConfig) => void;
  remove: () => void;
  refresh: () => void;
}) {
  const socialRecipients = useMemo(
    () => ({ telegram: channels.telegram.recipients, zalo: channels.zalo.recipients }),
    [channels],
  );
  const change = (mutator: (draft: NotificationRuleConfig) => void) => {
    const draft = clone(rule);
    mutator(draft);
    updateRule(draft);
  };
  const test = async () => {
    try {
      await axios.post(`notifications/rules/${rule.id}/test`);
      toast.success("Rule test queued");
      refresh();
    } catch {
      toast.error("No destination is ready");
    }
  };
  return (
    <div className="space-y-4 rounded-lg border border-secondary-variant p-4">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Input value={rule.id} onChange={(e) => change((draft) => (draft.id = e.target.value))} placeholder="Rule ID" />
        <Input value={rule.name} onChange={(e) => change((draft) => (draft.name = e.target.value))} placeholder="Name" />
        <select className="rounded-md border border-secondary-variant bg-background px-3" value={rule.event} onChange={(e) => change((draft) => (draft.event = e.target.value as NotificationEvent))}>
          {EVENTS.map((event) => <option key={event.value} value={event.value}>{event.label}</option>)}
        </select>
        <Input type="number" min={0} value={rule.cooldown} onChange={(e) => change((draft) => (draft.cooldown = Number(e.target.value)))} placeholder="Cooldown seconds" />
      </div>
      <FilterSwitch label="Rule enabled" isChecked={rule.enabled} onCheckedChange={(enabled) => change((draft) => (draft.enabled = enabled))} />
      <div>
        <div className="mb-2 text-sm font-medium">Cameras (none means all)</div>
        <div className="flex flex-wrap gap-3">
          {cameras.map((camera) => <FilterSwitch key={camera} label={camera} isChecked={rule.filters.cameras.includes(camera)} onCheckedChange={(checked) => change((draft) => { draft.filters.cameras = checked ? [...draft.filters.cameras, camera] : draft.filters.cameras.filter((value) => value !== camera); })} />)}
        </div>
      </div>
      {(rule.event === "alert" || rule.event === "object_detected") && (
        <div className="grid gap-3 md:grid-cols-2">
          <ListInput label="Labels" value={rule.filters.labels} onChange={(value) => change((draft) => (draft.filters.labels = value))} />
          <ListInput label="Zones" value={rule.filters.zones} onChange={(value) => change((draft) => (draft.filters.zones = value))} />
        </div>
      )}
      {rule.event === "face_recognized" && <ListInput label="Identities (* means all known; add unknown explicitly)" value={rule.filters.identities} onChange={(value) => change((draft) => (draft.filters.identities = value))} />}
      {rule.event === "semantic_trigger" && <ListInput label="Trigger names" value={rule.filters.trigger_names} onChange={(value) => change((draft) => (draft.filters.trigger_names = value))} />}
      {rule.event === "camera_monitoring" && <ListInput label="Conditions" value={rule.filters.conditions} onChange={(value) => change((draft) => (draft.filters.conditions = value))} />}
      <div>
        <div className="mb-2 text-sm font-medium">Destinations</div>
        <div className="space-y-2">
          <FilterSwitch label="WebPush registered devices" isChecked={rule.destinations.webpush} onCheckedChange={(checked) => change((draft) => (draft.destinations.webpush = checked))} />
          {(["telegram", "zalo"] as const).map((channel) => socialRecipients[channel].map((recipient) => (
            <FilterSwitch key={`${channel}-${recipient.id}`} label={`${channel}: ${recipient.name}`} isChecked={rule.destinations[channel].includes(recipient.id)} onCheckedChange={(checked) => change((draft) => { draft.destinations[channel] = checked ? [...draft.destinations[channel], recipient.id] : draft.destinations[channel].filter((id) => id !== recipient.id); })} />
          )))}
        </div>
      </div>
      <div className="flex gap-2">
        <Button size="sm" variant="outline" disabled={dirty || !rule.enabled} onClick={test}>Test saved rule</Button>
        <Button size="sm" variant="destructive" onClick={remove}>Remove rule</Button>
      </div>
    </div>
  );
}

function ListInput({ label, value, onChange }: { label: string; value: string[]; onChange: (value: string[]) => void }) {
  return (
    <label className="space-y-1 text-sm">
      <span>{label}</span>
      <Input value={value.join(", ")} onChange={(event) => onChange(event.target.value.split(",").map((item) => item.trim()).filter(Boolean))} placeholder="Comma-separated" />
    </label>
  );
}
