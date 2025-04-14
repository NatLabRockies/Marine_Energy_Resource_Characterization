# utils module - Part of tidal_visualizer package
import os
import ssl
import urllib3
import requests
import functools

import contextily as ctx
import numpy as np

from geopy.distance import geodesic


# def disable_ssl_verification():
#     # Disable SSL verification warnings
#     urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#
#     # Create a custom SSL context that doesn't verify certificates
#     ssl_ctx = ssl.create_default_context()
#     ssl_ctx.check_hostname = False
#     ssl_ctx.verify_mode = ssl.CERT_NONE
#
#     # Apply the custom SSL context to the requests library
#     old_merge_environment_settings = requests.Session.merge_environment_settings
#
#     def patch_merge_environment_settings(self, url, proxies, stream, verify, cert):
#         settings = old_merge_environment_settings(
#             self, url, proxies, stream, False, cert
#         )
#         return settings
#
#     # Apply the patch
#     requests.Session.merge_environment_settings = patch_merge_environment_settings
#
#     # Configure contextily cache directory
#     cache_dir = "/tmp/ctx_cache"
#     os.makedirs(cache_dir, exist_ok=True)
#     ctx.set_cache_dir(cache_dir)
#
#     # CRITICAL PATCH: Directly patch the request.get function used by contextily
#     original_get = requests.get
#
#     @functools.wraps(original_get)
#     def patched_get(url, **kwargs):
#         kwargs["verify"] = False
#         return original_get(url, **kwargs)
#
#     requests.get = patched_get
#
#     # Also patch the contextily _retryer function which calls requests.get
#     if hasattr(ctx.tile, "_retryer"):
#         original_retryer = ctx.tile._retryer
#
#         @functools.wraps(original_retryer)
#         def patched_retryer(tile_url, wait, max_retries):
#             session = requests.Session()
#             session.verify = False
#             # Use the same user agent that contextily uses
#             user_agent = getattr(ctx.tile, "USER_AGENT", "contextily")
#             request = session.get(tile_url, headers={"user-agent": user_agent})
#             request.raise_for_status()
#             return request.content
#
#         ctx.tile._retryer = patched_retryer
#
#     # For older versions of contextily that support this parameter:
#     if hasattr(ctx, "_ctx_pool_manager"):
#         ctx._ctx_pool_manager = urllib3.PoolManager(cert_reqs=ssl.CERT_NONE)


def disable_ssl_verification():
    # Disable SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Create a custom SSL context that doesn't verify certificates
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    # Apply the custom SSL context to the requests library
    old_merge_environment_settings = requests.Session.merge_environment_settings

    def patch_merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = old_merge_environment_settings(
            self, url, proxies, stream, False, cert
        )
        return settings

    # Apply the patch
    requests.Session.merge_environment_settings = patch_merge_environment_settings

    # Configure contextily to use our SSL context
    # ctx.set_cache_dir("/tmp/ctx_cache", create=True)  # Ensure cache dir exists
    ctx.set_cache_dir("/tmp/ctx_cache")  # Ensure cache dir exists

    # For older versions of contextily that support this parameter:
    ctx._ctx_pool_manager = urllib3.PoolManager(cert_reqs=ssl.CERT_NONE)
    # Disable SSL verification warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Configure requests to ignore SSL verification
    os.environ["CURL_CA_BUNDLE"] = ""  # This can help in some environments

    # Patch the underlying urllib3 pool manager used by requests and contextily
    original_init = urllib3.connectionpool.HTTPSConnectionPool.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["cert_reqs"] = "CERT_NONE"
        kwargs["assert_hostname"] = False
        original_init(self, *args, **kwargs)

    urllib3.connectionpool.HTTPSConnectionPool.__init__ = patched_init

    # Ensure contextily's cache directory exists
    # ctx.set_cache_dir("/tmp/ctx_cache", create=True)
    ctx.set_cache_dir("/tmp/ctx_cache")

    # For newer versions of contextily, directly patch the _retrieve_tile function
    # This is a more invasive approach but can be necessary
    try:
        original_retrieve = ctx.tile._retrieve_tile

        def patched_retrieve(url, *args, **kwargs):
            session = requests.Session()
            session.verify = False
            response = session.get(url, timeout=10)
            response.raise_for_status()
            return response.content

        # Patch the function if it exists
        if hasattr(ctx.tile, "_retrieve_tile"):
            ctx.tile._retrieve_tile = patched_retrieve
    except (AttributeError, ImportError):
        pass


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in meters."""
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def format_location_display(location_info):
    """Format location information for display."""
    display_name = location_info.get("location_display_name", "Unknown Location")
    state_abbr = location_info.get("state_abbr", "")

    if state_abbr:
        return f"{display_name}, {state_abbr}"
    else:
        return display_name


def format_area_of_interest_display(area_info):
    """Format area of interest information for display."""
    return area_info.get("display_name", "Unknown Area")


def format_combined_display(location_info, area_info):
    """Format combined location and area information for display."""
    location_display = format_location_display(location_info)
    area_display = format_area_of_interest_display(area_info)
    return f"{location_display} - {area_display}"


def calculate_bounds_center(bounds):
    """Calculate the center point of geographic bounds."""
    lon_min, lat_min, lon_max, lat_max = bounds
    return ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)


def calculate_power_statistics(power_data):
    """Calculate statistics for power density data."""
    stats = {
        "min": np.min(power_data),
        "max": np.max(power_data),
        "mean": np.mean(power_data),
        "median": np.median(power_data),
        "std": np.std(power_data),
        "p95": np.percentile(power_data, 95),
        "p99": np.percentile(power_data, 99),
    }
    return stats
