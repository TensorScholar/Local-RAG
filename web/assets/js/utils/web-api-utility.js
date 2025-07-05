/**
 * API Utility for Advanced RAG System Web Interface
 * 
 * This module provides a clean interface for communicating with the RAG system API,
 * handling request formatting, error management, and response processing.
 * 
 * @author Advanced RAG System Team
 * @version 1.0.0
 */

const api = (() => {
    // Base API URL - adjust if needed based on deployment
    const API_BASE_URL = '/api';
    
    /**
     * Generic API request handler with error management
     * 
     * @param {string} endpoint - API endpoint path
     * @param {Object} options - Fetch options
     * @returns {Promise<any>} - Parsed response data
     * @throws {Error} - On request failure
     */
    const request = async (endpoint, options = {}) => {
        try {
            const url = `${API_BASE_URL}/${endpoint.replace(/^\//, '')}`;
            
            const response = await fetch(url, {
                headers: {
                    'Accept': 'application/json',
                    ...(options.body instanceof FormData ? {} : {'Content-Type': 'application/json'}),
                    ...options.headers
                },
                ...options
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || `API request failed: ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    };
    
    return {
        /**
         * Get system health status
         * 
         * @returns {Promise<Object>} - System health information
         */
        getHealth: () => request('health'),
        
        /**
         * Get detailed system status
         * 
         * @returns {Promise<Object>} - Detailed system status and metrics
         */
        getSystemStatus: () => request('status'),
        
        /**
         * Submit a query to the RAG system
         * 
         * @param {Object} queryData - Query parameters
         * @returns {Promise<Object>} - Query response with generated text and metadata
         */
        submitQuery: (queryData) => request('query', {
            method: 'POST',
            body: JSON.stringify(queryData)
        }),
        
        /**
         * Upload a document to the RAG system
         * 
         * @param {File} file - Document file to upload
         * @param {Function} progressCallback - Optional callback for upload progress
         * @returns {Promise<Object>} - Upload result
         */
        uploadDocument: async (file, progressCallback = null) => {
            const formData = new FormData();
            formData.append('file', file);
            
            if (progressCallback) {
                // Create custom fetch with upload progress tracking
                return new Promise((resolve, reject) => {
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', `${API_BASE_URL}/documents/upload`);
                    
                    xhr.upload.addEventListener('progress', (event) => {
                        if (event.lengthComputable) {
                            const progress = Math.round((event.loaded / event.total) * 100);
                            progressCallback(progress);
                        }
                    });
                    
                    xhr.onload = () => {
                        if (xhr.status >= 200 && xhr.status < 300) {
                            resolve(JSON.parse(xhr.responseText));
                        } else {
                            try {
                                const errorData = JSON.parse(xhr.responseText);
                                reject(new Error(errorData.detail || `Upload failed: ${xhr.status}`));
                            } catch (e) {
                                reject(new Error(`Upload failed: ${xhr.status}`));
                            }
                        }
                    };
                    
                    xhr.onerror = () => reject(new Error('Upload failed due to network error'));
                    xhr.send(formData);
                });
            } else {
                // Use standard fetch without progress tracking
                return request('documents/upload', {
                    method: 'POST',
                    body: formData
                });
            }
        },
        
        /**
         * List all documents in the system
         * 
         * @returns {Promise<Object>} - List of indexed documents
         */
        listDocuments: () => request('documents'),
        
        /**
         * Delete a document from the system
         * 
         * @param {string} documentId - ID of document to delete
         * @returns {Promise<Object>} - Deletion result
         */
        deleteDocument: (documentId) => request(`documents/${documentId}`, {
            method: 'DELETE'
        }),
        
        /**
         * Clear the entire document index
         * 
         * @returns {Promise<Object>} - Clear index result
         */
        clearIndex: () => request('index/clear', {
            method: 'POST'
        })
    };
})();
