import type { SectionConfigOverrides } from "./types";

const notifications: SectionConfigOverrides = {
  base: {
    sectionDocs: "/configuration/notifications",
    hideSaveControls: true,
    restartRequired: [],
    fieldOrder: ["schema_version", "enabled", "channels", "rules", "delivery"],
    fieldGroups: {},
    hiddenFields: ["schema_version", "enabled", "channels", "rules", "delivery"],
    advancedFields: [],
  },
  global: {
    uiSchema: {
      "ui:before": { render: "NotificationsSettingsExtras" },
      schema_version: { "ui:widget": "hidden" },
      enabled: { "ui:widget": "hidden" },
      channels: { "ui:widget": "hidden" },
      rules: { "ui:widget": "hidden" },
      delivery: { "ui:widget": "hidden" },
    },
  },
  camera: {
    hiddenFields: ["enabled", "cooldown", "providers", "enabled_in_config"],
    fieldOrder: [],
    uiSchema: {},
  },
};

export default notifications;
