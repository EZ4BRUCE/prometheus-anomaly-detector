{{- define "formatMetricsList" -}}
{{- $metrics := .Values.prometheus.metricsList -}}
{{- $formatted := "" -}}
{{- range $index, $metric := $metrics -}}
  {{- if $index -}}
    {{- $formatted = printf "%s; %s" $formatted $metric -}}
  {{- else -}}
    {{- $formatted = $metric -}}
  {{- end -}}
{{- end -}}
{{- $formatted -}}
{{- end -}}