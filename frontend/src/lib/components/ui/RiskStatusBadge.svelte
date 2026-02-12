<script lang="ts">
	interface Props {
		status: string;
		size?: 'sm' | 'md' | 'lg';
	}

	let { status, size = 'md' }: Props = $props();

	const statusConfig = {
		HEALTHY: {
			color: 'green',
			icon: '✓',
			label: 'HEALTHY'
		},
		WARNING: {
			color: 'yellow',
			icon: '⚠',
			label: 'WARNING'
		},
		BREACH: {
			color: 'red',
			icon: '✗',
			label: 'BREACH'
		}
	} as const;

	const config = $derived(statusConfig[status as keyof typeof statusConfig] || statusConfig.HEALTHY);

	const sizeClasses = {
		sm: 'text-xs px-2 py-1',
		md: 'text-sm px-3 py-1.5',
		lg: 'text-base px-4 py-2'
	};

	const colorClasses = {
		green: 'bg-green-100 text-green-800 border-green-300',
		yellow: 'bg-yellow-100 text-yellow-800 border-yellow-300',
		red: 'bg-red-100 text-red-800 border-red-300'
	};
</script>

<span
	class="inline-flex items-center gap-1.5 font-semibold border rounded-md {sizeClasses[size]} {colorClasses[config.color]}"
>
	<span>{config.icon}</span>
	<span>{config.label}</span>
</span>
