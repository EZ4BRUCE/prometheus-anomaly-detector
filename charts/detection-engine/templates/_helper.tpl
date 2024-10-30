{{ define "configmap.data" }}
{{ .Values.detectionEngine | toYaml }}
{{ end }}

{{- define "configmap.data-digest" -}}
{{ include "configmap.data" . | sha256sum | trunc 8 }}
{{- end -}}
