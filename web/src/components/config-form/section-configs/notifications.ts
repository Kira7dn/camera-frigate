import type { SectionConfigOverrides } from "./types";

const notifications: SectionConfigOverrides = {
  base: {
    sectionDocs: "/configuration/notifications",
    restartRequired: [],
    fieldOrder: ["enabled", "email", "providers", "delivery"],
    fieldGroups: {},
    hiddenFields: ["enabled_in_config"],
    advancedFields: [],
  },
  global: {
    uiSchema: {
      "ui:before": { render: "NotificationsSettingsExtras" },
      enabled: { "ui:widget": "hidden" },
      email: { "ui:widget": "hidden" },
      cooldown: { "ui:widget": "hidden" },
      enabled_in_config: { "ui:widget": "hidden" },
    },
  },
  camera: {
    hiddenFields: ["enabled_in_config", "email"],
    fieldOrder: ["enabled", "cooldown", "providers"],
    uiSchema: {
      providers: { "ui:widget": "checkboxes" },
    },
  },
};

export default notifications;
