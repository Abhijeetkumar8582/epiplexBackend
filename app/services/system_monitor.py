"""
System Resource Monitoring Service
Tracks memory, CPU, disk I/O, and network usage
"""
import time
import psutil
from collections import deque
from typing import Dict, Optional
from threading import Lock
from datetime import datetime, timezone

from app.config import settings
from app.utils.logger import logger


class SystemMonitor:
    """
    Monitor system resources (memory, CPU, disk, network)
    Samples at regular intervals and maintains rolling window
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SystemMonitor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._lock = Lock()
        
        # Configuration
        self.enabled = getattr(settings, 'METRICS_ENABLED', True)
        self.sample_interval = getattr(settings, 'SYSTEM_MONITOR_INTERVAL_SECONDS', 30)
        self.retention_hours = getattr(settings, 'METRICS_RETENTION_HOURS', 24)
        self.max_samples = int((self.retention_hours * 3600) / self.sample_interval)
        
        # Storage: deque of (timestamp, metrics_dict)
        self._samples: deque = deque(maxlen=self.max_samples)
        
        # Process object
        self._process = psutil.Process()
        
        # Last sample time
        self._last_sample = 0
        
        # Start background monitoring if enabled
        if self.enabled:
            self._sample()  # Initial sample
    
    def _sample(self):
        """Sample current system metrics"""
        if not self.enabled:
            return
        
        try:
            timestamp = time.time()
            
            # Process memory
            process_memory = self._process.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)  # RSS in MB
            process_memory_vms_mb = process_memory.vms / (1024 * 1024)  # VMS in MB
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_memory_total_gb = system_memory.total / (1024 ** 3)
            system_memory_available_gb = system_memory.available / (1024 ** 3)
            system_memory_used_percent = system_memory.percent
            
            # Process CPU
            process_cpu_percent = self._process.cpu_percent(interval=0.1)
            
            # System CPU
            system_cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk I/O
            try:
                disk_io = psutil.disk_io_counters()
                disk_read_mb = disk_io.read_bytes / (1024 ** 2) if disk_io else 0
                disk_write_mb = disk_io.write_bytes / (1024 ** 2) if disk_io else 0
            except (AttributeError, PermissionError):
                disk_read_mb = 0
                disk_write_mb = 0
            
            # Network I/O (if available)
            try:
                network_io = psutil.net_io_counters()
                network_sent_mb = network_io.bytes_sent / (1024 ** 2) if network_io else 0
                network_recv_mb = network_io.bytes_recv / (1024 ** 2) if network_io else 0
            except (AttributeError, PermissionError):
                network_sent_mb = 0
                network_recv_mb = 0
            
            # Number of open file descriptors
            try:
                num_fds = self._process.num_fds() if hasattr(self._process, 'num_fds') else 0
            except (AttributeError, OSError):
                num_fds = 0
            
            # Number of threads
            try:
                num_threads = self._process.num_threads()
            except (AttributeError, OSError):
                num_threads = 0
            
            sample = {
                'timestamp': timestamp,
                'memory': {
                    'process_mb': round(process_memory_mb, 2),
                    'process_vms_mb': round(process_memory_vms_mb, 2),
                    'system_total_gb': round(system_memory_total_gb, 2),
                    'system_available_gb': round(system_memory_available_gb, 2),
                    'system_used_percent': round(system_memory_used_percent, 2)
                },
                'cpu': {
                    'process_percent': round(process_cpu_percent, 2),
                    'system_percent': round(system_cpu_percent, 2)
                },
                'disk_io': {
                    'read_mb': round(disk_read_mb, 2),
                    'write_mb': round(disk_write_mb, 2)
                },
                'network_io': {
                    'sent_mb': round(network_sent_mb, 2),
                    'recv_mb': round(network_recv_mb, 2)
                },
                'process': {
                    'num_fds': num_fds,
                    'num_threads': num_threads
                }
            }
            
            with self._lock:
                self._samples.append(sample)
                self._last_sample = timestamp
            
        except Exception as e:
            logger.warning("Failed to sample system metrics", error=str(e))
    
    def get_current_metrics(self) -> Dict:
        """Get the most recent system metrics"""
        if not self.enabled:
            return {}
        
        # Sample if needed
        current_time = time.time()
        if current_time - self._last_sample >= self.sample_interval:
            self._sample()
        
        with self._lock:
            if self._samples:
                return self._samples[-1].copy()
            else:
                # Return empty structure if no samples
                return {
                    'timestamp': time.time(),
                    'memory': {},
                    'cpu': {},
                    'disk_io': {},
                    'network_io': {},
                    'process': {}
                }
    
    def get_metrics_history(self, hours: Optional[float] = None) -> list:
        """
        Get metrics history
        
        Args:
            hours: Number of hours of history (default: all available)
        
        Returns:
            List of metric samples
        """
        if not self.enabled:
            return []
        
        cutoff_time = time.time() - (hours * 3600) if hours else 0
        
        with self._lock:
            if hours:
                return [
                    sample.copy() for sample in self._samples
                    if sample['timestamp'] >= cutoff_time
                ]
            else:
                return [sample.copy() for sample in self._samples]
    
    def get_average_metrics(self, hours: float = 1.0) -> Dict:
        """
        Get average metrics over specified time period
        
        Args:
            hours: Time period in hours
        
        Returns:
            Dictionary with average metrics
        """
        if not self.enabled:
            return {}
        
        history = self.get_metrics_history(hours=hours)
        if not history:
            return {}
        
        # Calculate averages
        n = len(history)
        
        avg_memory = {
            'process_mb': sum(s['memory'].get('process_mb', 0) for s in history) / n,
            'system_available_gb': sum(s['memory'].get('system_available_gb', 0) for s in history) / n,
            'system_used_percent': sum(s['memory'].get('system_used_percent', 0) for s in history) / n
        }
        
        avg_cpu = {
            'process_percent': sum(s['cpu'].get('process_percent', 0) for s in history) / n,
            'system_percent': sum(s['cpu'].get('system_percent', 0) for s in history) / n
        }
        
        return {
            'period_hours': hours,
            'sample_count': n,
            'memory': {k: round(v, 2) for k, v in avg_memory.items()},
            'cpu': {k: round(v, 2) for k, v in avg_cpu.items()}
        }
    
    def start_background_monitoring(self):
        """Start background monitoring thread (called from lifespan)"""
        if not self.enabled:
            return
        
        import asyncio
        import threading
        
        def monitor_loop():
            while True:
                time.sleep(self.sample_interval)
                self._sample()
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("System monitoring started", interval_seconds=self.sample_interval)


# Singleton instance
system_monitor = SystemMonitor()

