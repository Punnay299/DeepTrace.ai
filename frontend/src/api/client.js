const BASE_URL = "http://localhost:8000/api";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function fetchWrapper(endpoint, options = {}) {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
      },
    });

    const data = await response.json();

    if (!response.ok) {
      const msg = typeof data.detail === 'string'
        ? data.detail
        : data.detail?.detail || data.detail?.error || JSON.stringify(data.detail);
      throw new ApiError(
        msg || data.message || "An error occurred",
        response.status,
      );
    }

    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError("Failed to connect to the server", 0);
  }
}

export const apiClient = {
  uploadVideo: (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return fetchWrapper("/upload", {
      method: "POST",
      body: formData,
    });
  },

  checkStatus: (jobId) => {
    return fetchWrapper(`/status/${jobId}`);
  },

  getResults: (jobId) => {
    return fetchWrapper(`/result/${jobId}`);
  },
};
