/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, Modality } from '@google/genai';

// --- IndexedDB Helper Functions ---
const DB_NAME = 'photobookAiDB';
const DB_VERSION = 1;
const STORE_NAME = 'savedImages';

/**
 * Opens a connection to the IndexedDB database.
 * @returns A promise that resolves with the database connection.
 */
function openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onerror = () => reject(new Error(`IndexedDB error: ${request.error?.message}`));
        request.onsuccess = () => resolve(request.result);
        request.onupgradeneeded = (event) => {
            const db = (event.target as IDBOpenDBRequest).result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME, { keyPath: 'id' });
            }
        };
    });
}

/**
 * Adds an image to the IndexedDB.
 * @param image An object containing the image id and blob.
 * @returns A promise that resolves when the transaction is complete.
 */
async function addImageToDb(image: { id: string; blob: Blob }) {
    const db = await openDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.put(image);
    return new Promise<void>((resolve, reject) => {
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

/**
 * Retrieves all images from the IndexedDB.
 * @returns A promise that resolves with an array of image objects.
 */
async function getAllImagesFromDb(): Promise<{ id: string; blob: Blob }[]> {
    const db = await openDb();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();
    return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

/**
 * Deletes an image from the IndexedDB.
 * @param id The ID of the image to delete.
 * @returns A promise that resolves when the transaction is complete.
 */
async function deleteImageFromDb(id: string) {
    const db = await openDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.delete(id);
     return new Promise<void>((resolve, reject) => {
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

/**
 * Converts a data URL string to a Blob object.
 * @param dataurl The data URL string.
 * @returns The converted Blob.
 */
function dataURLtoBlob(dataurl: string): Blob {
    const arr = dataurl.split(',');
    const mimeMatch = arr[0].match(/:(.*?);/);
    if (!mimeMatch) {
        throw new Error('Invalid data URL format');
    }
    const mime = mimeMatch[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
}


// App state
let activeTab: 'generate' | 'background' | 'restore' | 'inpaint' = 'generate';
let uploadedFiles: { file: File; base64: string; id: string }[] = [];
let originalImage: { file: File; base64: string; id: string } | null = null;
let backgroundImage: { file: File; base64: string; id: string } | null = null;
let restorationImage: { file: File; base64: string; id: string } | null = null;
let inpaintImage: { file: File; base64: string; id: string } | null = null;
let isLoading = false;
let lastGeneratedImageUrl: string | null = null;
let lastUsedPrompt: string | null = null;
// savedImages now holds object URLs, not data URLs, for better performance
let savedImages: { url: string; id: string }[] = [];


// DOM Element references
// These will be re-assigned when tabs are switched
let imageUploadEl: HTMLElement | null;
let imagePreviewEl: HTMLElement | null;
let promptInputEl: HTMLTextAreaElement | null;
let generateBtnEl: HTMLButtonElement | null;
let fileInputEl: HTMLInputElement | null;
let analyzeBtnEl: HTMLButtonElement | null;
let aspectRatioEl: HTMLSelectElement | null;
let clearPromptBtnEl: HTMLButtonElement | null;
let keepFaceCheckboxEl: HTMLInputElement | null;
let keepFaceSectionEl: HTMLElement | null;

// Inpaint state variables
let inpaintDisplayCanvas: HTMLCanvasElement | null = null;
let inpaintMaskCanvas: HTMLCanvasElement | null = null;
let inpaintOriginalImage: HTMLImageElement | null = null;
let inpaintIsDrawing = false;
let inpaintBrushSize = 30;
let inpaintUndoStack: ImageData[] = [];


// Common elements
let outputEl: HTMLElement;
let zoomModalEl: HTMLElement;
let zoomedImgEl: HTMLImageElement;
let modalCloseBtnEl: HTMLElement;
let tabContentEl: HTMLElement;
let savedImagesGridEl: HTMLElement;


/**
 * Initializes the application, sets up the DOM, and attaches event listeners.
 */
function App() {
    document.body.innerHTML = `
        <main>
            <div class="column column-left">
                <div class="input-section">
                    <div class="header-section">
                        <img src="https://www.photobookvietnam.net/images/logo_rev.png" alt="Photobook Vietnam Logo" class="header-logo" />
                    </div>
                    <div class="tabs" style="display: none">
                        <button id="tab-generate" class="tab-button active" data-tab="generate">Tạo ảnh AI</button>
                        <button id="tab-background" class="tab-button" data-tab="background">Thay nền ảnh</button>
                        <button id="tab-restore" class="tab-button" data-tab="restore">Phục hồi ảnh cũ</button>
                        <button id="tab-inpaint" class="tab-button" data-tab="inpaint">Inpaint</button>
                    </div>
                    <div id="tab-content"></div>
                </div>
            </div>
            <div class="column column-right">
                <section id="output" class="output-section" aria-live="polite">
                    <div class="placeholder">
                        <p>Ảnh của bạn sẽ xuất hiện ở đây.</p>
                    </div>
                </section>
                <section id="saved-images-section" class="saved-images-section">
                    <h2>Ảnh đã lưu</h2>
                    <div id="saved-images-grid">
                        <p class="placeholder-text">Chưa có ảnh nào được lưu.</p>
                    </div>
                </section>
            </div>
        </main>
        <div id="zoom-modal" class="modal">
            <span id="modal-close-btn" class="modal-close" aria-label="Đóng chế độ xem phóng to">&times;</span>
            <img class="modal-content" id="zoomed-img" alt="Ảnh đã phóng to">
        </div>
    `;

    // Get references to common DOM elements
    outputEl = document.getElementById('output')!;
    zoomModalEl = document.getElementById('zoom-modal')!;
    zoomedImgEl = document.getElementById('zoomed-img') as HTMLImageElement;
    modalCloseBtnEl = document.getElementById('modal-close-btn')!;
    tabContentEl = document.getElementById('tab-content')!;
    savedImagesGridEl = document.getElementById('saved-images-grid')!;

    // Tab switching logic
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', (e) => {
            const target = e.currentTarget as HTMLButtonElement;
            const tabName = target.dataset.tab as 'generate' | 'background' | 'restore' | 'inpaint';
            if (tabName !== activeTab) {
                activeTab = tabName;
                document.querySelector('.tab-button.active')?.classList.remove('active');
                target.classList.add('active');
                renderActiveTab();
            }
        });
    });

    // Modal listeners
    modalCloseBtnEl.addEventListener('click', closeZoomModal);
    zoomModalEl.addEventListener('click', (event) => {
        if (event.target === zoomModalEl) {
            closeZoomModal();
        }
    });
    
    // Handle responsive layout
    window.addEventListener('resize', handleLayoutChange);
    handleLayoutChange(); // Initial check

    // Load saved images from IndexedDB asynchronously
    getAllImagesFromDb().then(imagesFromDb => {
        // Revoke any existing object URLs to prevent memory leaks
        savedImages.forEach(img => URL.revokeObjectURL(img.url));
        
        savedImages = imagesFromDb.map(imgData => ({
            id: imgData.id,
            url: URL.createObjectURL(imgData.blob),
        }));
        renderSavedImages(); // Render images once they are loaded
    }).catch(e => {
        console.error("Failed to load images from IndexedDB", e);
        savedImages = [];
        renderSavedImages(); // Render empty state on error
    });

    renderActiveTab(); // Render the initial active tab
    // Initial render will be empty, then populated by the async call above.
    renderSavedImages(); 
}

/**
* Handles responsive layout changes between desktop and mobile.
*/
function handleLayoutChange() {
    const mainEl = document.querySelector('main');
    const leftColInputSection = document.querySelector('.input-section');
    const rightCol = document.querySelector('.column-right');
    const outputSection = document.getElementById('output');
    const savedImagesSection = document.getElementById('saved-images-section');
    
    if (!mainEl || !leftColInputSection || !rightCol || !outputSection || !savedImagesSection) return;

    const isMobile = window.innerWidth < 1024;

    if (isMobile) {
        // Mobile layout: Move results to a collapsible panel at the bottom of the left column
        let mobileWrapper = document.getElementById('mobile-results-wrapper');
        if (!mobileWrapper) {
            mobileWrapper = document.createElement('div');
            mobileWrapper.id = 'mobile-results-wrapper';
            mobileWrapper.innerHTML = `
                <div class="mobile-results-header">
                    <span>Xem kết quả & Ảnh đã lưu</span>
                    <span class="toggle-arrow"></span>
                </div>
                <div class="mobile-results-content"></div>
            `;
            const header = mobileWrapper.querySelector('.mobile-results-header');
            header?.addEventListener('click', () => {
                 mobileWrapper?.classList.toggle('collapsed');
            });
        }
        
        // Move the sections into the mobile wrapper's content area
        const mobileContent = mobileWrapper.querySelector('.mobile-results-content');
        if (mobileContent) {
            mobileContent.appendChild(outputSection);
            mobileContent.appendChild(savedImagesSection);
        }

        // Add the wrapper to the BOTTOM of the input section
        leftColInputSection.appendChild(mobileWrapper);
        mobileWrapper.classList.remove('collapsed'); // Expand by default on mobile

    } else {
        // Desktop layout: Move results back to the right column
        let mobileWrapper = document.getElementById('mobile-results-wrapper');
        if (mobileWrapper) {
            rightCol.appendChild(outputSection);
            rightCol.appendChild(savedImagesSection);
            mobileWrapper.remove();
        }
    }
}


/**
 * Renders the UI for the currently active tab.
 */
function renderActiveTab() {
    if (activeTab === 'generate') {
        renderGeneratorUI();
    } else if (activeTab === 'background') {
        renderBackgroundChangerUI();
    } else if (activeTab === 'restore') {
        renderRestorationUI();
    } else {
        renderInpaintUI();
    }
}

/**
 * Renders the UI for the AI Image Generator tab.
 */
function renderGeneratorUI() {
    tabContentEl.innerHTML = `
        <div class="generator-top-container">
            <div id="image-upload" class="drop-zone large" role="button" tabindex="0" aria-label="Vùng tải ảnh lên">
                <div id="generator-image-preview" class="single-preview-container large">
                    <p style="font-size: 14pt">Nhấn(Bấm) ở đây để chọn ảnh</p>
                </div>
                <input type="file" id="file-input" accept="image/*" hidden>
            </div>
            <div class="generator-options">
                <div id="keep-face-section" class="checkbox-section hidden">
                    <input type="checkbox" checked="true" id="keep-face-checkbox">
                    <label for="keep-face-checkbox">Giữ khuôn mặt</label>
                </div>
                <div id="aspect-ratio-wrapper" class="aspect-ratio-section">
                    <label for="aspect-ratio">Tỷ lệ khung hình</label>
                    <select id="aspect-ratio">
                        <option value="1:1" selected>Vuông (1:1)</option>
                        <option value="3:4">Dọc (3:4)</option>
                        <option value="4:3">Ngang (4:3)</option>
                        <option value="9:16">Cao (9:16)</option>
                        <option value="16:9">Rộng (16:9)</option>
                    </select>
                </div>
            </div>
        </div>

        <div style="display: none">
            <div class="prompt-header">
                <label id="prompt-input-label" for="prompt-input">Mô tả</label>
                <div class="prompt-buttons">
                    <button id="clear-prompt-btn" aria-label="Xóa mô tả">Xóa</button>
                    <button id="analyze-btn">Phân tích ảnh</button>
                </div>
            </div>
            <textarea id="prompt-input" placeholder="Ví dụ: Một chú mèo oai vệ đội mũ dự tiệc" aria-labelledby="prompt-input-label"></textarea>
        </div>

        <div class="sub-tabs">
            <button class="sub-tab-button active" data-subtab="baby">Bé dưới 1 tuồi</button>
            <button class="sub-tab-button" data-subtab="so-sinh">Bé sơ sinh</button>
            <button class="sub-tab-button" data-subtab="trung-thu">Trung Thu</button>
        </div>

        <div id="sub-tab-content">
            <div id="sub-tab-panel-so-sinh" class="sub-tab-panel">
                <div class="templates-section">
                    <label>Sơ sinh</label>
                    <div class="template-buttons-container">
                        <button id="baby-template-btn-1" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby1.jpg" alt="">
                            <span>Baby 1</span>
                        </button>
                        <button id="baby-template-btn-2" class="template-button" aria-label="Sử dụng mẫu Trung Thu 2">
                            <img src="https://canvasvietnam.com/images/baby2.jpg" alt="">
                            <span>Baby 2</span>
                        </button>
                        <button id="baby-template-btn-3" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/baby3.jpg" alt="">
                            <span>Baby 3</span>
                        </button>
                        <button id="baby-template-btn-4" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/baby4.jpg" alt="">
                            <span>Baby 4</span>
                        </button>
                        <button id="baby-template-btn-5" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/baby5.jpg" alt="">
                            <span>Baby 5</span>
                        </button>
                        <button id="baby-template-btn-6" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby6.jpg" alt="">
                            <span>Baby 6</span>
                        </button>
                        <button id="baby-template-btn-7" class="template-button" aria-label="Sử dụng mẫu Trung Thu 2">
                            <img src="https://canvasvietnam.com/images/baby7.jpg" alt="">
                            <span>Baby 7</span>
                        </button>
                        <button id="baby-template-btn-8" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/baby8.jpg" alt="">
                            <span>Baby 8</span>
                        </button>
                        <button id="baby-template-btn-9" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/baby9.jpg" alt="">
                            <span>Baby 9</span>
                        </button>
                        <button id="baby-template-btn-10" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby10.jpg" alt="">
                            <span>Baby 10</span>
                        </button>
                        <button id="baby-template-btn-11" class="template-button" aria-label="Sử dụng mẫu Trung Thu 2">
                            <img src="https://canvasvietnam.com/images/baby11.jpg" alt="">
                            <span>Baby 11</span>
                        </button>
                        <button id="baby-template-btn-12" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/baby12.jpg" alt="">
                            <span>Baby 12</span>
                        </button>
                    </div>
                </div>
            </div>
            <div id="sub-tab-panel-trung-thu" class="sub-tab-panel">
                <div class="templates-section">
                    <label>Bé trai</label>
                    <div class="template-buttons-container">
                        <button id="trung-thu-template-btn-1" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/trungthu1.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>Dưới 1 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-2" class="template-button" aria-label="Sử dụng mẫu Trung Thu 2">
                            <img src="https://canvasvietnam.com/images/trungthu2.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>1-3 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-3" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu3.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>1-3 tuổi</span>
                        </button>

                        <button id="trung-thu-template-btn-6" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu6.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>1-3 tuổi</span>
                        </button>
                    </div>
                </div>

                <div class="templates-section">
                    <label>Bé gái</label>
                    <div class="template-buttons-container">
                        
                        <button id="trung-thu-template-btn-4" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu4.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>Dưới 1 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-5" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu5.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>3-5 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-7" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu7.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>5-10 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-8" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu8.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>3-5 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-9" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu9.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>5-10 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-10" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu10.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>5-10 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-11" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu11.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>5-10 tuổi</span>
                        </button>
                        <button id="trung-thu-template-btn-12" class="template-button" aria-label="Sử dụng mẫu Trung Thu 3">
                            <img src="https://canvasvietnam.com/images/trungthu12.jpg" alt="Bé trai mặc đồ Trung Thu, xung quanh là đèn lồng và đầu lân">
                            <span>1-3 tuổi</span>
                        </button>
                    </div>
                </div>
            </div>
             <div id="sub-tab-panel-baby" class="sub-tab-panel active">
                <div class="templates-section">
                    <label>Gần 1 tuổi</label>
                    <div class="template-buttons-container">
                        <button id="baby-template-btn-13" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby13.jpg" alt="">
                            <span>Baby 13</span>
                        </button>
                        <button id="baby-template-btn-14" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby14.jpg" alt="">
                            <span>Baby 14</span>
                        </button>
                        <button id="baby-template-btn-15" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby15.jpg" alt="">
                            <span>Baby 15</span>
                        </button>
                        <button id="baby-template-btn-16" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby16.jpg" alt="">
                            <span>Baby 16</span>
                        </button>
                        <button id="baby-template-btn-17" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby17.jpg" alt="">
                            <span>Baby 17</span>
                        </button>
                        <button id="baby-template-btn-18" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby18.jpg" alt="">
                            <span>Baby 18</span>
                        </button>
                        <button id="baby-template-btn-19" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby19.jpg" alt="">
                            <span>Baby 19</span>
                        </button>
                        <button id="baby-template-btn-20" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby20.jpg" alt="">
                            <span>Baby 20</span>
                        </button>
                        <button id="baby-template-btn-21" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby21.jpg" alt="">
                            <span>Baby 21</span>
                        </button>
                        <button id="baby-template-btn-22" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby22.jpg" alt="">
                            <span>Baby 22</span>
                        </button>
                        <button id="baby-template-btn-23" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby23.jpg" alt="">
                            <span>Baby 23</span>
                        </button>
                        <button id="baby-template-btn-24" class="template-button" aria-label="Sử dụng mẫu Trung Thu 1">
                            <img src="https://canvasvietnam.com/images/baby24.jpg" alt="">
                            <span>Baby 24</span>
                        </button>
                    </div>
                </div>
                
                
            </div>
        </div>

        <button id="generate-btn" disabled>Tạo ảnh</button>
    `;

    // Get references to DOM elements for this tab
    imageUploadEl = document.getElementById('image-upload')!;
    imagePreviewEl = document.getElementById('generator-image-preview')!;
    promptInputEl = document.getElementById('prompt-input') as HTMLTextAreaElement;
    generateBtnEl = document.getElementById('generate-btn') as HTMLButtonElement;
    fileInputEl = document.getElementById('file-input') as HTMLInputElement;
    analyzeBtnEl = document.getElementById('analyze-btn') as HTMLButtonElement;
    aspectRatioEl = document.getElementById('aspect-ratio') as HTMLSelectElement;
    clearPromptBtnEl = document.getElementById('clear-prompt-btn') as HTMLButtonElement;
    keepFaceCheckboxEl = document.getElementById('keep-face-checkbox') as HTMLInputElement;
    keepFaceSectionEl = document.getElementById('keep-face-section') as HTMLElement;
    const trungThuTemplateBtn1 = document.getElementById('trung-thu-template-btn-1')!;
    const trungThuTemplateBtn2 = document.getElementById('trung-thu-template-btn-2')!;
    const trungThuTemplateBtn3 = document.getElementById('trung-thu-template-btn-3')!;
    const trungThuTemplateBtn4 = document.getElementById('trung-thu-template-btn-4')!;
    const trungThuTemplateBtn5 = document.getElementById('trung-thu-template-btn-5')!;
    const trungThuTemplateBtn6 = document.getElementById('trung-thu-template-btn-6')!;
    const trungThuTemplateBtn7 = document.getElementById('trung-thu-template-btn-7')!;
    const trungThuTemplateBtn8 = document.getElementById('trung-thu-template-btn-8')!;
    const trungThuTemplateBtn9 = document.getElementById('trung-thu-template-btn-9')!;
    const trungThuTemplateBtn10 = document.getElementById('trung-thu-template-btn-10')!;
    const trungThuTemplateBtn11 = document.getElementById('trung-thu-template-btn-11')!;
    const trungThuTemplateBtn12 = document.getElementById('trung-thu-template-btn-12')!;

 	const babyTemplateBtn1 = document.getElementById('baby-template-btn-1')!;
	const babyTemplateBtn2 = document.getElementById('baby-template-btn-2')!;
	const babyTemplateBtn3 = document.getElementById('baby-template-btn-3')!;
	const babyTemplateBtn4 = document.getElementById('baby-template-btn-4')!;
	const babyTemplateBtn5 = document.getElementById('baby-template-btn-5')!;
	const babyTemplateBtn6 = document.getElementById('baby-template-btn-6')!;
	const babyTemplateBtn7 = document.getElementById('baby-template-btn-7')!;
	const babyTemplateBtn8 = document.getElementById('baby-template-btn-8')!;
	const babyTemplateBtn9 = document.getElementById('baby-template-btn-9')!;
	const babyTemplateBtn10 = document.getElementById('baby-template-btn-10')!;
	const babyTemplateBtn11 = document.getElementById('baby-template-btn-11')!;
	const babyTemplateBtn12 = document.getElementById('baby-template-btn-12')!;
	const babyTemplateBtn13 = document.getElementById('baby-template-btn-13')!;
	const babyTemplateBtn14 = document.getElementById('baby-template-btn-14')!;
	const babyTemplateBtn15 = document.getElementById('baby-template-btn-15')!;
	const babyTemplateBtn16 = document.getElementById('baby-template-btn-16')!;
	const babyTemplateBtn17 = document.getElementById('baby-template-btn-17')!;
	const babyTemplateBtn18 = document.getElementById('baby-template-btn-18')!;
	const babyTemplateBtn19 = document.getElementById('baby-template-btn-19')!;
	const babyTemplateBtn20 = document.getElementById('baby-template-btn-20')!;
	const babyTemplateBtn21 = document.getElementById('baby-template-btn-21')!;
	const babyTemplateBtn22 = document.getElementById('baby-template-btn-22')!;
	const babyTemplateBtn23 = document.getElementById('baby-template-btn-23')!;
	const babyTemplateBtn24 = document.getElementById('baby-template-btn-24')!;


    // Attach event listeners for this tab
    imageUploadEl.addEventListener('click', () => fileInputEl!.click());
    fileInputEl.addEventListener('change', handleFileSelect);
    imageUploadEl.addEventListener('dragover', handleDragOver);
    imageUploadEl.addEventListener('dragleave', handleDragLeave);
    imageUploadEl.addEventListener('drop', handleDrop);
    promptInputEl.addEventListener('input', updateGenerateButtonState);
    generateBtnEl.addEventListener('click', handleGenerateClick);
    analyzeBtnEl.addEventListener('click', handleAnalyzeClick);
    clearPromptBtnEl.addEventListener('click', handleClearPromptClick);

    // New Sub-tab logic
    document.querySelectorAll('.sub-tab-button').forEach(button => {
        button.addEventListener('click', (e) => {
            const target = e.currentTarget as HTMLButtonElement;
            const subtabName = target.dataset.subtab;

            // Update button active state
            document.querySelector('.sub-tab-button.active')?.classList.remove('active');
            target.classList.add('active');

            // Update panel visibility
            document.querySelectorAll('.sub-tab-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            document.getElementById(`sub-tab-panel-${subtabName}`)?.classList.add('active');
        });
    });

    trungThuTemplateBtn1.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an image of a baby (1–2 years old) in the atmosphere of the Mid-Autumn Festival. The baby is sitting on a rustic bamboo bed, wearing a traditional Vietnamese red yếm (halter-neck top) paired with green pants, along with a vibrant lion dance hat decorated with fluffy white trim and colorful embroidery. The baby is holding a small mooncake in both hands, gazing at it with curious, innocent eyes. Around the scene are iconic Mid-Autumn details: a lion dance head placed nearby, a plate of ripe golden persimmons, a black lacquer tray with a teapot and lotus flowers, and vases with blooming lotuses and kumquat branches. The bamboo wall backdrop is decorated with red and green star-shaped lanterns, colorful fabric lanterns, and shimmering tinsel garlands, recreating a lively festival setting. Warm lighting highlights the scene, evoking a cozy and nostalgic feeling of traditional Vietnamese Mid-Autumn celebrations. High-resolution, ultra-realistic, natural, festive, and heartwarming style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });
    
    trungThuTemplateBtn2.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A full-body portrait of a young boy sitting on a small bamboo bench, set in a vibrant Mid-Autumn Festival scene. He is wearing a light beige traditional Vietnamese ao dai with simple brown trousers and a classic black khan xep (turban-style headpiece). In his hand, he holds a red paper fan and smiles brightly.
Next to him is a small bamboo table displaying traditional mooncakes and an old-style oil lamp. Surrounding him are festive Mid-Autumn decorations: a large red carp-shaped lion head, star-shaped lanterns, bamboo toys, and fresh green bamboo leaves.
The background features a red wooden wall adorned with hanging festive ornaments, illuminated by warm indoor lighting that creates a nostalgic and culturally rich atmosphere.
Style: Cinematic, ultra-realistic photography with strong traditional Vietnamese cultural aesthetics and vibrant festive details.
Lighting: Warm indoor festival lighting with golden tones and soft shadows.
Mood: Joyful, festive, cultural, and nostalgic.
Camera Settings:
Shot: Full-body portrait, eye-level perspective
Focus: Strong focus on the boy’s face (must match the uploaded reference image exactly), with the background softly blurred.
Aspect Ratio: 9:16
Important Notes:
Face: Use the exact face, hairstyle, expression, and features from the uploaded photo. Absolutely no modification allowed.
Outfit:
Beige traditional ao dai
Simple brown trousers
Classic black khan xep (turban-style headpiece)
Accessory: Red paper fan in hand
Background: Include a large red carp lion head, star lanterns, bamboo toys, mooncakes, an oil lamp, green bamboo leaves, and a red wooden wall.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });

    trungThuTemplateBtn3.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A full-body portrait of a young boy in a Mid-Autumn Festival setting, dressed in a traditional Vietnamese festive outfit. He is wearing a white cross-collar shirt with a lion head embroidery on the chest, paired with bright red shorts and a matching red cap. He is barefoot, smiling happily, and playfully raising one hand. Next to him is a large red festival drum and scattered traditional toys. The background features a giant decorative red carp (symbol of Mid-Autumn), red cherry blossom branches, colorful butterfly lanterns, and festive ornaments. The atmosphere is joyful, festive, and culturally rich, styled in a cinematic, ultra-realistic way. IMPORTANT: Preserve the exact face, identity, and expression from the uploaded reference photo. The face must match perfectly and not be altered`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });

    trungThuTemplateBtn4.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an image of a 1-year-old baby girl in the cheerful atmosphere of the Mid-Autumn Festival. The baby is sitting happily on the floor, wearing a traditional Vietnamese red silk halter top (áo yếm) paired with green pants, with a matching red bow in her hair. She is holding a small wooden drumstick, joyfully playing the red toy drum in front of her, with an excited, bright smile.
The background is decorated with vibrant Mid-Autumn details: red paper lanterns, a colorful lion dance head, and a playful cutout of a lion figure. Behind her, a festive banner with the words “Tết Trung Thu” is displayed. Warm golden lighting enhances the vivid red and green tones, creating a lively, traditional, and heartwarming celebration scene. High-resolution, ultra-realistic, festive, and adorable style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });

    trungThuTemplateBtn5.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A full-body portrait of a young child dressed in a traditional red festive outfit for Mid-Autumn Festival. The outfit is a sleeveless red halter-top with a decorative floral emblem at the chest, paired with a flowing red skirt layered with white fabric underneath. The child wears a matching headpiece with blue and red accents, smiling with joy. She is standing in front of a vibrant Mid-Autumn Festival stage decorated with a large glowing full moon at the center, surrounded by red curtains and large yellow-orange flower decorations. Traditional items such as drums, star-shaped lanterns, and a colorful lion dance mask are placed around the stage. The atmosphere is festive, warm, and joyful, styled in a cinematic and ultra-realistic way. IMPORTANT: Preserve the exact face, identity, and expression from the uploaded reference photo. The face must remain identical to the uploaded image, with no alterations`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    trungThuTemplateBtn6.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A traditional Mid-Autumn Festival portrait of a young child sitting on a wooden chair in a festive indoor setup. The child is dressed in a simple brown áo bà ba outfit with buttoned top and loose pants, paired with a black-and-white checkered khăn rằn tied around the head. On the child's lap is a colorful star lantern decorated with green tinsel. Around the setting are typical Mid-Autumn decorations: red velvet curtain backdrop with multiple star lanterns hanging, painted paper masks with smiling faces, a small lion dance head, and dried floral arrangements with warm autumn tones. On a low wooden table next to the child are mooncakes, a ceramic tea set, toy figurines, and festive ornaments. The atmosphere is warm, nostalgic, and joyful. Keep everything exactly the same, only replace the face with the exact face from the uploaded image, preserving all natural features and expression accurately.",
"style": "Studio photography, cultural, realistic",
"mood": "Festive, nostalgic, warm, joyful",
"lighting": "Soft warm studio light, highlighting the child and decorations, gentle shadows",
"details": {
"subject": {
"face": "Use the exact face from the uploaded image",
"outfit": "Brown áo bà ba with buttoned shirt and loose pants",
"accessories": "Black-and-white checkered khăn rằn tied on head",
"pose": "Sitting on wooden chair, holding a star lantern on lap"
},
"props": {
"lantern": "Star lantern with green tinsel",
"decor": "Red velvet curtain with star lanterns, painted masks, lion dance head",
"table": "Wooden table with mooncakes, tea set, and Mid-Autumn ornaments"
},
"background": "Festive indoor Mid-Autumn setup with red curtains and seasonal decor"`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });



    trungThuTemplateBtn7.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A festive Mid-Autumn portrait of a little girl sitting gracefully against a deep red background (use upload photo). She has a slim, delicate figure and is dressed in a red halter-neck dress with layered fabric and a flowing white skirt underneath. A matching red headband adorns her hair. She leans her head gently against an oversized mooncake prop with intricate traditional patterns, as if peacefully resting. Behind her, decorative golden paper fish float in the air, symbolizing prosperity, while a large red flower and golden lattice panel accentuate the scene. The overall setup is artistic, warm, and celebratory of Mid-Autumn traditions. Lighting is soft and even, casting a festive glow over the child and props. Keep all outfits, background, and details the same. Only replace the face with the exact face from the uploaded image, preserving`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });



    trungThuTemplateBtn8.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A little girl (use upload photo) in a red traditional festive outfit is standing barefoot on a giant mooncake stage. She is gracefully raising her right arm, holding a small lantern, with her left arm extended sideways like dancing. Her outfit is a deep red halter dress with layered fabric and a large bow on the chest, decorated with subtle golden patterns. Her hair is styled neatly with a red flower accessory. The background is a Mid-Autumn Festival setup with a big glowing moon, bright red velvet curtains, colorful paper clouds, green fabric ribbons, and traditional festive toys (lion head, paper rooster, mini lanterns). The atmosphere is joyful, vibrant, and theatrical. Keep all details exactly the same, only replace the face with the on`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    trungThuTemplateBtn9.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A high-resolution studio portrait capturing the joyful spirit of a young Vietnamese girl, aged approximately 8-10, celebrating the Mid-Autumn Festival. The girl is seated cross-legged on a richly patterned red and gold oriental rug, facing forward with a beaming, and look directed at the viewer. She is holding up two elaborately patterned, golden-brown mooncakes, one in each hand. Her attire consists of a vibrant red, traditional Chinese-style silk top with a halter neckline and delicate, shiny gold dragonfly embroidery throughout. She also wears a dark teal or navy blue pleated skirt. Her hair is styled in two high pigtail buns, adorned with large red satin bows and ribbons that cascade down her back. Her eyes are bright, and she has subtle, festive orange eyeshadow and red lipstick. The background is a deep, rich red, pleated velvet curtain, creating a luxurious and festive ambiance. Centered above the girl is a large, glowing, soft yellow full moon prop. To her left, partially obscured, stands a detailed, fluffy red lion dance head decoration with golden eyes, resting on a dark wooden pedestal. A colorful pinwheel toy and a small red drum are also visible near the lion head. To her right, several decorative paper fish lanterns with intricate patterns and vibrant colors (red, gold, green, blue) hang from the curtain, alongside two smaller, traditional red paper lanterns. A low, dark wooden table or stool is positioned behind her to the left, holding a traditional ceramic tea set (teapot and cups) and a red tinsel garland. Another small, woven bamboo basket containing additional mooncakes sits on the rug near her feet. The lighting is soft, warm, and inviting, emanating from the front and slightly above, highlighting the girl's face and the mooncakes while casting gentle shadows on the curtains, enhancing the festive and cozy atmosphere. The overall composition is balanced and full of traditional cultural elements, rich in red, gold, and warm tones, conveying celebration and happiness.
`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });



    trungThuTemplateBtn10.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A full-body studio photograph featuring a young girl (from the input image), she holding a small woven basket containing a miniature mooncake. The child is dressed in a flowing, traditional-inspired dress with layers of pastel peach, mint green, and light blue fabrics, complemented by a white long-sleeved undershirt with ruffled cuffs. The lighting is high-key and soft, creating a dreamlike and ethereal atmosphere.
The background is an elaborately decorated Mid-Autumn Festival themed set. Dominating the center-back is a large, circular, golden-orange full moon backdrop with the English calligraphy "Trung Thu" (Mid-Autumn) written on it. Above the moon, a vibrant red and yellow decorative fish lantern hangs. To the left, a tall cream-colored arched panel features an ornate cutout shape enclosing a decorative white lantern. Lush green lotus leaves and white lotus flowers, some adorned with subtle fairy lights, are scattered across the scene, adding depth and organic elements.
On the ground, wispy white fog or mist covers the floor, creating a magical, cloud-like effect. To the left, a large, intricately carved wooden prop resembling a gigantic mooncake sits next to a small, glossy red drum-like stool. To the right of the child, a small wooden table holds a traditional woven teapot and several individual mooncakes. Further to the right, another cream-colored arched panel frames a hanging lantern, and stylized dark green and blue mountain props create a layered landscape. White rabbit cutouts are visible near the child's table and further back, symbolizing the Mid-Autumn legend.
The overall aesthetic is whimsical and enchanting, with a rich but gentle color palette dominated by pastels (peach, mint, cream, light blue) and warm accents (golden orange, crimson red, brown), offset by deep teal and dark green in the background elements. The composition is balanced, with the child slightly off-center, framed by the detailed and festive decor, creating a sense of wonder and celebration.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    trungThuTemplateBtn11.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. A whimsical and joyful photograph of a young Vietnamese child, in 6-8 years old, dressed in a beautiful traditional Chinese Hanfu-style outfit. The outfit features flowing, sheer fabrics in soft pastel green and creamy yellow tones, with intricate, delicate embroidery or patterns adorning the bodice and a wide sash at the waist. The child wears a charming white and pink bunny ear headband, and their hair is styled in a traditional updo with small floral decorations. They hold a glowing, spherical yellow object, reminiscent of a lantern or a full moon, in their right hand, raised slightly. A sweet, innocent smile lights up their face, conveying joy and warmth.
The background is a soft, seamless pastel green, creating a bright and airy atmosphere. Several bokeh-effect, blurry spherical yellow and orange lights (suggesting moons, lanterns, or auspicious fruits) float gently in the distant background, adding a dreamlike quality. Subtle, ethereal floral motifs also appear softly blurred in the background.
The lighting is soft, ethereal, and even, casting a gentle glow on the subject without harsh shadows, characteristic of high-key studio photography. The dominant color palette consists of light greens, creams, soft yellows, with touches of pale orange, pink, and white.
The composition is a three-quarter length shot, centered on the child, allowing for ample negative space that enhances the whimsical and traditional aesthetic. The overall style is reminiscent of fine art photography, celebrating cultural tradition with a dreamy, festive, and innocent charm, perfect for a Mid-Autumn Festival theme.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    trungThuTemplateBtn12.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an image of a 1-year-old baby girl in the cheerful atmosphere of the Mid-Autumn Festival. The baby is sitting happily on the floor, wearing a traditional Vietnamese red silk halter top (áo yếm) paired with green pants, with a matching red bow in her hair. She is holding a small wooden drumstick, joyfully playing the red toy drum in front of her, with an excited, bright smile.
The background is decorated with vibrant Mid-Autumn details: red paper lanterns, a colorful lion dance head, and a playful cutout of a lion figure. Behind her, a festive banner with the words “Tết Trung Thu” is displayed. Warm golden lighting enhances the vivid red and green tones, creating a lively, traditional, and heartwarming celebration scene. High-resolution, ultra-realistic, festive, and adorable style`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });




    babyTemplateBtn1.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create a newborn photoshoot of a 1–2 month-old baby sleeping peacefully on a fluffy white rug, dressed in a hand-knitted green dinosaur outfit with a hood featuring cute spikes. Surround the baby with small crocheted dinosaur plush toys and pastel dinosaur eggs placed neatly on green tropical leaves. The warm bokeh background creates a dreamy, fairytale-like atmosphere. Soft studio lighting highlights the baby’s delicate skin and the handmade textures, resulting in a cozy, whimsical, and adorable scene`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });

    babyTemplateBtn2.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create a heartwarming newborn photoshoot of a 1–2 month-old baby peacefully sleeping on top of a soft pastel green long-neck dinosaur plush toy. The baby is dressed in a cream-colored knitted romper with long sleeves and a matching knitted hat with tiny ears. The baby’s body rests gently on the dinosaur’s back, arms folded under the cheek, creating a serene and dreamy sleeping pose. The background is a smooth pastel mint green, minimal and clean, enhancing the softness and purity of the scene. Soft studio lighting highlights the baby’s delicate skin and the plush dinosaur’s gentle texture, resulting in an adorable, whimsical, and tender atmosphere`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn3.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an adorable newborn photoshoot of a baby dressed in a fluffy brown bear costume with little ears and a tiny bee decoration on the hood. The baby is gently leaning on a plush honey pot labeled “SUNNY,” surrounded by soft orange flowers and small plush bees. Warm orange pastel backdrop with soft glowing lighting, dreamy and heartwarming atmosphere, high-resolution studio photography, clean and minimal composition.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn4.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an image of the baby in the uploaded photo, preserving all natural features and expressions. The baby is sitting against a bright yellow background, wearing a yellow-and-white gingham bonnet tied under the chin, with a lemon placed on top of the head. The outfit is a sleeveless yellow bodysuit with a small embroidered duck on the front. The baby’s cheeks are chubby and slightly blushed. High-resolution, ultra-realistic, studio-quality photo with vibrant colors and a cheerful atmosphere.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn5.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create a heartwarming studio photoshoot of a newborn baby lying peacefully on the tummy with arms gently folded under the chin, dressed in a fluffy brown bear costume with a tiny bee on the hood. The baby is positioned between two plush honey jars dripping with felt honey, surrounded by orange and white felt flowers and small plush bees. Warm orange pastel backdrop, soft glowing light, dreamy and adorable atmosphere, high-resolution, clean and minimal newborn photography style`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn6.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create a newborn photoshoot of a 1–2 month-old baby sleeping peacefully on a fluffy white rug, dressed in a hand-knitted green dinosaur outfit with a hood featuring cute spikes. Surround the baby with small crocheted dinosaur plush toys and pastel dinosaur eggs placed neatly on green tropical leaves. The warm bokeh background creates a dreamy, fairytale-like atmosphere. Soft studio lighting highlights the baby’s delicate skin and the handmade textures, resulting in a cozy, whimsical, and adorable scene.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn7.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an newborn photo of a baby, peacefully sleeping while wrapped snugly in a deep red swaddle blanket. The baby is wearing a festive lion dance hat made of soft white fabric with fluffy details, colorful embroidery, and playful pom-poms, evoking a cheerful Mid-Autumn Festival vibe. The baby is placed gently in a rustic wooden bowl lined with soft fur, resting on a woven mat with warm earthy tones. Around the setup, add traditional Mid-Autumn toys such as a small fabric lion toy and festive decorations. The background is wooden flooring, creating a cozy, rustic atmosphere. High-resolution, ultra-realistic, soft studio lighting, warm and heartwarming style`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn8.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create a professional studio newborn photograph of a sleeping baby, The baby is dressed in a Winnie the Pooh-themed costume: a mustard yellow knit beanie with small rounded bear ears, a classic red knit long-sleeved top, and mustard yellow knit pants.
The baby is posed peacefully, curled up on its stomach with its head gently resting on its hands, eyes closed in deep sleep. It lies within a miniature dark brown wooden bed prop, which is lined with a thick, creamy white, chunky cable-knit blanket that softly cradles the baby.
The foreground features a rustic dark wooden floor with visible grain, subtly adorned with scattered green foliage, including ivy vines and eucalyptus leaves. A small, soft Winnie the Pooh plush toy (yellow with a red shirt) sits upright on the wooden floor in the lower right corner, facing the viewer.
The background is softly blurred with warm, earthy tones, suggesting a cozy, rustic indoor setting. It includes indistinct dark brown wooden elements, a hint of a blurred green potted plant on the left, and a soft, light-colored blurred textile on the right.
The lighting is soft, warm, and diffused studio light, creating gentle highlights and shadows that emphasize the baby's features and the textures of the knitwear. The composition is a full shot, captured from a slightly elevated perspective looking down, with a shallow depth of field. The baby, bed, and immediate foreground elements are in sharp focus, while the background is beautifully out of focus, enhancing the subject. The overall aesthetic is whimsical, cozy, adorable, and heartwarming, characteristic of high-quality newborn photography. The dominant color palette consists of warm yellows (mustard), classic Pooh red, creamy whites, rich browns, and subtle greens.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn9.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an newborn photo of the baby, preserving all natural features and expressions. The baby is peacefully sleeping, dressed in a full knitted brown bodysuit with a matching hat featuring small horns and ears, resembling a baby calf. The baby is lying on a chunky knitted brown blanket, curled up comfortably. Next to the baby is a small wooden stool with a glass bottle of milk, and on the floor a tiny felt cow toy with a miniature wooden bucket. On the right side, add a bundle of golden rice stalks for decoration. The background is warm brown tones with a rustic, cozy farm-like atmosphere. High-resolution, ultra-realistic, soft studio lighting, dreamy and heartwarming style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn10.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an newborn photo of the baby, preserving all natural features and expressions. The baby is peacefully sleeping, dressed in a full knitted brown bodysuit with a matching hat featuring small horns and ears, resembling a baby calf. The baby is lying on a chunky knitted brown blanket, curled up comfortably. Next to the baby is a small wooden stool with a glass bottle of milk, and on the floor a tiny felt cow toy with a miniature wooden bucket. On the right side, add a bundle of golden rice stalks for decoration. The background is warm brown tones with a rustic, cozy farm-like atmosphere. High-resolution, ultra-realistic, soft studio lighting, dreamy and heartwarming style`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn11.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an newborn photo of the baby girl, preserving all natural features and expressions. The baby is peacefully sleeping, dressed in a delicate white lace dress with a soft tutu skirt. She wears a pink floral headband on her head, adding a sweet and dreamy touch. The baby is cuddling against a large plush white unicorn with yarn mane, lying comfortably as if hugging it. Around her is a sheer fabric decorated with golden stars and sequins, creating a magical, fairy-tale atmosphere. The background is soft beige, enhancing the gentle, elegant, and enchanting vibe. High-resolution, ultra-realistic, soft studio lighting, dreamy and heartwarming style`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });



    babyTemplateBtn12.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. Create an newborn photo of the baby girl, preserving all natural features and expressions. The baby is peacefully sleeping on her side in a curled-up pose, wearing a delicate white lace romper. On her head is a golden lace crown, giving a princess-like appearance. She rests on a soft cream-colored textured blanket, blending harmoniously with the neutral background. The baby’s cheeks are round and slightly blushed, with a serene, angelic sleeping expression. High-resolution, ultra-realistic, soft studio lighting, dreamy and elegant style`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn13.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo, natural baby skin and soft features. She is lying down comfortably on a soft pastel milestone blanket decorated with baby girl elements. She wears a cute pastel-colored baby dress with a matching headband. Beside her is a small decorated cake with a candle. The background and blanket setup include pastel pink balloons, teddy bears, and flowers. Her name “Happy Birthday” is beautifully integrated into the backdrop design (like on a milestone board, balloon letters, or signage), making it look like a real milestone event photoshoot. High-resolution, photorealistic, professional baby milestone theme.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn14.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is lying on her back (not face-down), comfortably positioned in the center on a soft pastel milestone blanket. She is dressed in a baby Cinderella-inspired outfit — a light blue princess-style romper or dress with tiny puff sleeves, soft tutu skirt, and a matching headband or small tiara. Beside her is a small decorated cake with a candle. The background and props are Cinderella-themed: pastel blue balloons, a magical carriage design, castle silhouette, sparkling stars, and fairy-tale decorations. The backdrop also beautifully integrates her name “Happy Birthday”as part of the design (like in balloon letters, a castle banner, or sparkling signage). Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone poster style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn15.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is lying on her back (not face-down), comfortably positioned in the center on a soft milestone blanket. She is dressed in a baby Ariel-inspired outfit — a mermaid-style romper or dress in turquoise and purple, with seashell details and a matching headband or tiny tiara. Beside her is a decorated cake topped with a candle. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Little Mermaid themed: under-the-sea details, seashells, starfish, corals, ocean waves, and fairy-tale elements. Add cute themed stuffed props near her (like a plush Flounder, Sebastian, or seashell pillow) to enhance the Ariel theme. Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn16.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is lying on her back (not face-down), comfortably positioned in the center on a soft milestone blanket. She is dressed in a baby Belle-inspired outfit — a golden-yellow princess-style romper or dress with soft ruffles and a matching headband or tiny tiara. Beside her is a decorated cake with a candle. The background and props are Beauty and the Beast themed: an enchanted castle backdrop, roses, books, candelabra, and other magical fairy-tale decorations. Add cute themed stuffed props next to her (like a soft plush rose, Mrs. Potts and Chip-inspired toys, or a mini enchanted rose in a glass dome) to match the Belle theme. The backdrop also beautifully integrates her name “Happy Birthday” as part of the design (like a royal banner, glowing signage, or castle arch). Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone poster style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn17.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is lying on her back (not face-down), comfortably positioned in the center on a soft milestone blanket. She is dressed in a baby Snow White-inspired outfit — a blue and yellow princess-style dress with puffed sleeves, red accents, and a matching red headband or bow. She is also wearing cute red baby shoes to complete the Snow White look. Beside her is a decorated cake topped with a candle to mark the month. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Snow White themed: enchanted forest details, apples, a small wooden cottage, and fairy-tale elements. Add cute themed stuffed props near her (like a plush red apple, forest animals, or a tiny magic mirror) to enhance the Snow White theme. Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn18.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is lying on her back (not face-down), comfortably positioned in the center on a soft milestone blanket. She is dressed in a baby Moana-inspired outfit — a cute island-style romper or dress in red and beige with Polynesian-inspired patterns, plus a matching flower crown or tropical headband. She is also wearing tiny baby slippers or soft shoes that complement the Moana theme. Beside her is a decorated cake topped with a candle to mark the month. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Moana themed: tropical island details, the ocean, palm trees, corals, seashells, and fairy-tale elements. Add cute themed stuffed props near her (like a plush Hei Hei chicken, Pua the pig, or tropical flowers) to enhance the Moana theme. Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn19.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is sitting upright comfortably in the center on a soft milestone blanket. She is dressed in a baby Frozen-inspired outfit — an Elsa-style light blue princess dress with sparkly details, flowing tulle skirt, and shimmering cape. She has Frozen-inspired hair styled like a soft baby version of Elsa’s braid, with tiny snowflake accents. She is also wearing tiny silver or light blue baby shoes to complete the Frozen look. Beside her is a decorated cake topped with a candle to mark the month. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Frozen themed: snowy mountains, snowflakes, icicles, and a magical ice castle design. Add cute themed stuffed props near her (like a plush Olaf snowman or snowflake pillows) to enhance the Frozen theme. Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn20.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is sitting upright comfortably in the center on a soft milestone blanket. She is dressed in a baby Minnie Mouse-inspired outfit — a red polka dot dress with puffed sleeves, white gloves-style detail, and a matching Minnie Mouse headband with black ears and a big red bow. She is also wearing tiny red or black baby shoes to complete the Minnie look. Beside her is a decorated cake topped with a candle to mark the month. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday”” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Minnie Mouse themed: pink and red balloons, polka dot patterns, hearts, and Minnie Mouse plush toys. Add cute themed stuffed props near her (like a plush Minnie Mouse, Mickey toy, or little polka-dot gift boxes) to enhance the Minnie theme. Soft, cheerful lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn21.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is sitting upright comfortably in the center on a soft milestone blanket. She is dressed in a cute baby Santa Claus-inspired outfit — a red and white Christmas dress with fluffy white trim, a matching Santa hat or festive headband, and tiny red baby shoes to complete the look. Beside her is a decorated Christmas cake topped with a candle to mark the month. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Christmas themed: decorated Christmas tree, wrapped gifts, stockings, snowflakes, candy canes, and fairy lights. Add cute themed stuffed props near her (like a plush reindeer, snowman, or Christmas teddy bear) to enhance the holiday theme. Soft, warm festive lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn22.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is sitting upright comfortably in the center on a soft milestone blanket. She is dressed in a baby Aurora (Sleeping Beauty)-inspired outfit — a soft pink princess gown with elegant details, puffy skirt, and a tiny golden crown or tiara on her head. She is also wearing cute pink baby shoes to complete the Aurora princess look. Beside her is a decorated cake topped with a candle to mark the month. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Sleeping Beauty themed: a magical castle, enchanted forest, fairies’ sparkles, and dreamy floral decorations. Add cute themed stuffed props near her (like a plush spinning wheel, tiny fairy dolls, or a soft pink pillow) to enhance the Aurora theme. Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    babyTemplateBtn23.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic milestone portrait of a baby girl with the exact real face from the uploaded photo. She is sitting upright comfortably in the center on a soft milestone blanket. She is dressed in a baby Tinker Bell-inspired outfit — a sparkly green fairy dress with ruffled skirt, tiny fairy wings on her back, and a matching green headband or bun hairstyle with a little ribbon. She is also wearing cute green or silver baby shoes to complete the fairy look. Beside her is a decorated cake topped with a candle to mark the month. A wooden display board is placed next to her, showing the milestone greeting “Happy Birthday” in realistic letterboard style, just like an authentic milestone photoshoot setup. The background and props are Tinker Bell themed: magical forest details, glowing fairy dust sparkles, mushrooms, flowers, and tiny lanterns. Add cute themed stuffed props near her (like a plush fairy wand, mini Peter Pan hat, or small forest animals) to enhance the fairy vibe. Soft, magical lighting, natural baby skin tones, high-resolution, photorealistic, professional milestone photoshoot style.`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });

    


    babyTemplateBtn24.addEventListener('click', () => {
        if (promptInputEl) {
            const prompt = `Ultra-realistic birthday portrait of a baby girl with the exact real face from the uploaded photo. She is sitting upright in the center, wearing a puffy pink princess dress with rosette details, sparkly bodice, and a big bow accent. She has long wavy black hair with a black ribbon headband. Around her is an enchanted forest theme setup: giant colorful butterflies, glowing fairy lights, lush flowers in pink, purple, and cream, and wooden log props. Behind her is a dreamy forest backdrop with tall trees and twinkling golden lights like fireflies. In front of her are large white marquee letters spelling out “ONE” to mark her first birthday. Add soft, whimsical lighting, magical atmosphere, and high-resolution detail for a professional studio photoshoot style`;
            promptInputEl.value = prompt;
            promptInputEl.dispatchEvent(new Event('input')); // To update button state
        }
    });


    // Add logic for highlighting template buttons
    document.querySelectorAll('.template-button').forEach(button => {
        button.addEventListener('click', () => {
            // Remove 'selected' from all other buttons
            document.querySelectorAll('.template-button').forEach(otherButton => {
                otherButton.classList.remove('selected');
            });
            // Add 'selected' to the clicked button
            button.classList.add('selected');
        });
    });

    // Restore state
    renderImagePreviews();
    updateGenerateButtonState();
}


/**
 * Renders the UI for the Background Changer tab.
 */
function renderBackgroundChangerUI() {
    tabContentEl.innerHTML = `
        <div class="upload-group">
            <label>Ảnh gốc (để giữ lại chủ thể)</label>
            <div id="original-image-upload" class="drop-zone small" role="button" tabindex="0">
                <div id="original-image-preview" class="single-preview-container">
                    <p>Tải ảnh gốc</p>
                </div>
                <input type="file" id="original-file-input" accept="image/*" hidden>
            </div>
        </div>

        <div class="upload-group">
            <label>Ảnh nền (tùy chọn)</label>
            <div id="background-image-upload" class="drop-zone small" role="button" tabindex="0">
                 <div id="background-image-preview" class="single-preview-container">
                    <p>Tải ảnh nền</p>
                </div>
                <input type="file" id="background-file-input" accept="image/*" hidden>
            </div>
        </div>
        
        <div class="prompt-header">
            <label for="background-prompt-input">Mô tả nền (nếu không tải ảnh nền)</label>
            <div class="prompt-buttons">
                 <button id="generate-bg-prompt-btn">Tạo mô tả</button>
                 <div id="suggestion-wrapper" class="suggestion-wrapper hidden">
                    <input type="text" id="suggestion-input" placeholder="Gợi ý nền, ví dụ: trong rừng">
                    <button id="confirm-suggestion-btn" aria-label="Xác nhận gợi ý">✓</button>
                    <button id="cancel-suggestion-btn" aria-label="Hủy bỏ">✕</button>
                 </div>
            </div>
        </div>
        <textarea id="background-prompt-input" placeholder="Ví dụ: Một bãi biển nhiệt đới lúc hoàng hôn"></textarea>
        
        <button id="change-background-btn" disabled>Thay nền</button>
    `;

    // Get references and attach listeners
    const originalUploadEl = document.getElementById('original-image-upload')!;
    const originalFileInputEl = document.getElementById('original-file-input') as HTMLInputElement;
    const backgroundUploadEl = document.getElementById('background-image-upload')!;
    const backgroundFileInputEl = document.getElementById('background-file-input') as HTMLInputElement;
    const backgroundPromptInputEl = document.getElementById('background-prompt-input') as HTMLTextAreaElement;
    const generateBgPromptBtnEl = document.getElementById('generate-bg-prompt-btn') as HTMLButtonElement;
    const changeBackgroundBtnEl = document.getElementById('change-background-btn') as HTMLButtonElement;

    // Suggestion UI elements
    const suggestionWrapperEl = document.getElementById('suggestion-wrapper')!;
    const suggestionInputEl = document.getElementById('suggestion-input') as HTMLInputElement;
    const confirmSuggestionBtnEl = document.getElementById('confirm-suggestion-btn')!;
    const cancelSuggestionBtnEl = document.getElementById('cancel-suggestion-btn')!;

    originalUploadEl.addEventListener('click', () => originalFileInputEl.click());
    originalFileInputEl.addEventListener('change', (e) => handleSingleFile(e, 'original'));
    originalUploadEl.addEventListener('dragover', handleDragOver);
    originalUploadEl.addEventListener('dragleave', handleDragLeave);
    originalUploadEl.addEventListener('drop', (e) => handleSingleFileDrop(e, 'original'));

    backgroundUploadEl.addEventListener('click', () => backgroundFileInputEl.click());
    backgroundFileInputEl.addEventListener('change', (e) => handleSingleFile(e, 'background'));
    backgroundUploadEl.addEventListener('dragover', handleDragOver);
    backgroundUploadEl.addEventListener('dragleave', handleDragLeave);
    backgroundUploadEl.addEventListener('drop', (e) => handleSingleFileDrop(e, 'background'));
    
    [backgroundPromptInputEl, originalFileInputEl, backgroundFileInputEl].forEach(el => {
        el.addEventListener('input', updateChangeBackgroundBtnState);
        el.addEventListener('change', updateChangeBackgroundBtnState);
    });

    changeBackgroundBtnEl.addEventListener('click', handleChangeBackgroundClick);

    // Suggestion UI Listeners
    generateBgPromptBtnEl.addEventListener('click', () => {
        if (!originalImage) {
            alert("Vui lòng tải lên ảnh gốc trước.");
            return;
        }
        generateBgPromptBtnEl.classList.add('hidden');
        suggestionWrapperEl.classList.remove('hidden');
        suggestionInputEl.focus();
    });

    cancelSuggestionBtnEl.addEventListener('click', () => {
        suggestionWrapperEl.classList.add('hidden');
        generateBgPromptBtnEl.classList.remove('hidden');
        suggestionInputEl.value = '';
    });

    confirmSuggestionBtnEl.addEventListener('click', () => {
        handleGenerateBackgroundPromptClick(suggestionInputEl.value);
    });
    suggestionInputEl.addEventListener('keydown', (e) => {
        if(e.key === 'Enter') {
            handleGenerateBackgroundPromptClick(suggestionInputEl.value);
        }
    });

    // Restore state
    renderSingleImagePreview('original');
    renderSingleImagePreview('background');
    updateChangeBackgroundBtnState();
}

/**
 * Renders the UI for the Photo Restoration tab.
 */
function renderRestorationUI() {
    tabContentEl.innerHTML = `
        <div class="upload-group">
            <label>Tải ảnh cũ, hư hại</label>
            <div id="restoration-image-upload" class="drop-zone" role="button" tabindex="0">
                <div id="restoration-image-preview" class="single-preview-container large">
                    <p>Tải ảnh để phục hồi</p>
                </div>
                <input type="file" id="restoration-file-input" accept="image/*" hidden>
            </div>
        </div>

        <div class="options-grid">
             <div class="checkbox-section">
                <input type="checkbox" id="colorize-checkbox">
                <label for="colorize-checkbox">Tô màu</label>
            </div>
             <div class="checkbox-section">
                <input type="checkbox" id="face-rotate-checkbox">
                <label for="face-rotate-checkbox">Xoay mặt (chính diện)</label>
            </div>
        </div>

        <div class="outfit-section">
            <label for="outfit-select">Thay trang phục (tùy chọn)</label>
            <div class="outfit-controls">
                 <select id="outfit-select">
                    <option value="none">Không thay đổi</option>
                    <option value="Vest nam">Vest nam</option>
                    <option value="Vest nữ">Vest nữ</option>
                    <option value="Áo dài Việt Nam">Áo dài Việt Nam</option>
                    <option value="Áo sơ mi">Áo sơ mi</option>
                    <option value="custom">Khác (nhập bên dưới)</option>
                </select>
                <textarea id="custom-outfit-input" class="hidden" placeholder="Mô tả trang phục bạn muốn..."></textarea>
            </div>
        </div>
        
        <button id="restore-btn" disabled>Phục hồi ảnh</button>
    `;
    
    // Get references and attach listeners
    const restorationUploadEl = document.getElementById('restoration-image-upload')!;
    const restorationFileInputEl = document.getElementById('restoration-file-input') as HTMLInputElement;
    const restoreBtnEl = document.getElementById('restore-btn') as HTMLButtonElement;
    const outfitSelectEl = document.getElementById('outfit-select') as HTMLSelectElement;
    const customOutfitInputEl = document.getElementById('custom-outfit-input') as HTMLTextAreaElement;

    restorationUploadEl.addEventListener('click', () => restorationFileInputEl.click());
    restorationFileInputEl.addEventListener('change', (e) => handleSingleFile(e, 'restoration'));
    restorationUploadEl.addEventListener('dragover', handleDragOver);
    restorationUploadEl.addEventListener('dragleave', handleDragLeave);
    restorationUploadEl.addEventListener('drop', (e) => handleSingleFileDrop(e, 'restoration'));
    
    restoreBtnEl.addEventListener('click', handleRestoreClick);
    
    outfitSelectEl.addEventListener('change', () => {
        if (outfitSelectEl.value === 'custom') {
            customOutfitInputEl.classList.remove('hidden');
        } else {
            customOutfitInputEl.classList.add('hidden');
        }
    });

    // Restore state
    renderSingleImagePreview('restoration');
    updateRestoreButtonState();
}

/**
 * Renders the UI for the new Inpaint tab.
 */
function renderInpaintUI() {
    tabContentEl.innerHTML = `
        <div class="upload-group">
            <label>Tải ảnh để chỉnh sửa Inpaint</label>
            <div id="inpaint-image-upload" class="drop-zone large" role="button" tabindex="0">
                <div id="inpaint-image-preview" class="single-preview-container large">
                     <p>Tải ảnh lên</p>
                </div>
                <input type="file" id="inpaint-file-input" accept="image/*" hidden>
            </div>
        </div>
        
        <div id="inpaint-canvas-wrapper" class="hidden">
            <canvas id="inpaint-display-canvas"></canvas>
        </div>

        <div id="inpaint-tools" class="inpaint-tools-container hidden">
            <div class="tool">
                <label for="brush-size">Cỡ cọ: <span id="brush-size-value">${inpaintBrushSize}</span></label>
                <input type="range" id="brush-size" min="5" max="100" value="${inpaintBrushSize}" step="1">
            </div>
            <div class="tool-buttons">
                <button id="undo-btn" class="action-btn small">Hoàn tác</button>
                <button id="clear-mask-btn" class="action-btn small">Xóa vùng chọn</button>
            </div>
        </div>

        <div class="prompt-header">
            <label for="inpaint-prompt-input">Mô tả vùng cần chỉnh sửa</label>
        </div>
        <textarea id="inpaint-prompt-input" placeholder="Ví dụ: Thêm một chiếc mũ cao bồi"></textarea>
        
        <button id="inpaint-btn" disabled>Chỉnh sửa ảnh</button>
    `;

    // Get references
    const inpaintUploadEl = document.getElementById('inpaint-image-upload')!;
    const inpaintFileInputEl = document.getElementById('inpaint-file-input') as HTMLInputElement;
    const inpaintBtnEl = document.getElementById('inpaint-btn') as HTMLButtonElement;
    const inpaintPromptEl = document.getElementById('inpaint-prompt-input') as HTMLTextAreaElement;
    
    inpaintDisplayCanvas = document.getElementById('inpaint-display-canvas') as HTMLCanvasElement;
    
    // Attach listeners
    inpaintUploadEl.addEventListener('click', () => inpaintFileInputEl.click());
    inpaintFileInputEl.addEventListener('change', (e) => handleSingleFile(e, 'inpaint'));
    inpaintUploadEl.addEventListener('dragover', handleDragOver);
    inpaintUploadEl.addEventListener('dragleave', handleDragLeave);
    inpaintUploadEl.addEventListener('drop', (e) => handleSingleFileDrop(e, 'inpaint'));
    inpaintBtnEl.addEventListener('click', handleInpaintClick);
    inpaintPromptEl.addEventListener('input', updateInpaintButtonState);

    // Restore state
    if (inpaintImage) {
        initializeInpaintCanvas(inpaintImage);
    }
    updateInpaintButtonState();
}

/**
 * Scrolls the view to the results section.
 */
function scrollToResults() {
    const isMobile = window.innerWidth < 1024;
    let targetElement;

    if (isMobile) {
        targetElement = document.getElementById('mobile-results-wrapper');
    } else {
        targetElement = document.getElementById('output');
    }

    if (targetElement) {
        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Processes a single file for upload, used by both click and drag-and-drop.
 */
async function processSingleFile(file: File, type: 'original' | 'background' | 'restoration' | 'inpaint') {
    if (!file || !file.type.startsWith('image/')) {
        return;
    }

    const base64 = await fileToBase64(file);
    const id = `file-${Date.now()}`;
    const fileData = { file, base64, id };

    if (type === 'original') {
        originalImage = fileData;
    } else if (type === 'background') {
        backgroundImage = fileData;
    } else if (type === 'restoration') {
        restorationImage = fileData;
    } else if (type === 'inpaint') {
        inpaintImage = fileData;
        initializeInpaintCanvas(fileData);
    }
    
    if (type !== 'inpaint') {
      renderSingleImagePreview(type);
    }
    
    if (type === 'original' || type === 'background') {
        updateChangeBackgroundBtnState();
    } else if (type === 'restoration') {
        updateRestoreButtonState();
    } else {
        updateInpaintButtonState();
    }
}

/**
 * Handles single file upload for various tabs from a file input.
 */
async function handleSingleFile(event: Event, type: 'original' | 'background' | 'restoration' | 'inpaint') {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
        await processSingleFile(file, type);
    }
}

/**
 * Handles single file drag-and-drop for various tabs.
 */
async function handleSingleFileDrop(event: DragEvent, type: 'original' | 'background' | 'restoration' | 'inpaint') {
    event.preventDefault();
    (event.currentTarget as HTMLElement)?.classList.remove('active');
    if (event.dataTransfer?.files) {
        const file = event.dataTransfer.files[0];
        if (file) {
            await processSingleFile(file, type);
        }
    }
}


/**
 * Renders preview for single image upload zones.
 */
function renderSingleImagePreview(type: 'original' | 'background' | 'restoration') {
    const containerId = `${type}-image-preview`;
    const containerEl = document.getElementById(containerId);
    if (!containerEl) return;

    let imageData;
    let placeholderText = '';
    switch (type) {
        case 'original':
            imageData = originalImage;
            placeholderText = 'Tải ảnh gốc';
            break;
        case 'background':
            imageData = backgroundImage;
            placeholderText = 'Tải ảnh nền';
            break;
        case 'restoration':
            imageData = restorationImage;
            placeholderText = 'Tải ảnh để phục hồi';
            break;
    }

    if (imageData) {
        containerEl.innerHTML = `
            <img src="${imageData.base64}" alt="${imageData.file.name}" class="single-preview-img"/>
            <button class="remove-btn" data-type="${type}" aria-label="Remove image">&times;</button>
        `;
        containerEl.querySelector('.remove-btn')?.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent parent drop-zone click event
            if (type === 'original') originalImage = null;
            else if (type === 'background') backgroundImage = null;
            else if (type === 'restoration') restorationImage = null;

            renderSingleImagePreview(type);
             if (type === 'original' || type === 'background') {
                updateChangeBackgroundBtnState();
            } else {
                updateRestoreButtonState();
            }
        });
    } else {
        containerEl.innerHTML = `<p>${placeholderText}</p>`;
    }
}


/**
 * Updates the state of the "Change Background" button.
 */
function updateChangeBackgroundBtnState() {
    const btn = document.getElementById('change-background-btn') as HTMLButtonElement | null;
    const promptEl = document.getElementById('background-prompt-input') as HTMLTextAreaElement | null;
    if (!btn || !promptEl) return;

    // Logic: Disable prompt if background image exists
    if (backgroundImage) {
        promptEl.disabled = true;
        promptEl.value = '';
    } else {
        promptEl.disabled = false;
    }
    
    const hasOriginal = !!originalImage;
    const hasBackground = !!backgroundImage;
    const hasPrompt = promptEl.value.trim() !== '';

    btn.disabled = isLoading || !hasOriginal || (!hasBackground && !hasPrompt);
}

/**
 * Updates the state of the "Restore Photo" button.
 */
function updateRestoreButtonState() {
     const btn = document.getElementById('restore-btn') as HTMLButtonElement | null;
     if (!btn) return;
     btn.disabled = isLoading || !restorationImage;
}

/**
 * Updates the state of the "Inpaint" button.
 */
function updateInpaintButtonState() {
    const btn = document.getElementById('inpaint-btn') as HTMLButtonElement | null;
    const promptEl = document.getElementById('inpaint-prompt-input') as HTMLTextAreaElement | null;
    if (!btn || !promptEl || !inpaintMaskCanvas) return;

    const hasImage = !!inpaintImage;
    const hasPrompt = promptEl.value.trim() !== '';

    // Check if anything has been drawn on the mask
    const maskCtx = inpaintMaskCanvas.getContext('2d');
    const maskData = maskCtx?.getImageData(0, 0, inpaintMaskCanvas.width, inpaintMaskCanvas.height).data;
    let hasDrawing = false;
    if (maskData) {
        for (let i = 3; i < maskData.length; i += 4) {
            if (maskData[i] > 0) { // Check alpha channel
                hasDrawing = true;
                break;
            }
        }
    }

    btn.disabled = isLoading || !hasImage || !hasPrompt || !hasDrawing;
}

/**
 * Handles generating a background prompt suggestion.
 */
async function handleGenerateBackgroundPromptClick(suggestion: string = "") {
    if (!originalImage) {
        alert("Vui lòng tải lên ảnh gốc trước.");
        return;
    }

    isLoading = true;
    const generateBtn = document.getElementById('generate-bg-prompt-btn') as HTMLButtonElement;
    const suggestionWrapperEl = document.getElementById('suggestion-wrapper')!;
    const promptButtonsEl = generateBtn.parentElement!;

    suggestionWrapperEl.classList.add('hidden');
    const spinner = document.createElement('span');
    spinner.className = 'small-spinner';
    promptButtonsEl.appendChild(spinner);
    updateChangeBackgroundBtnState();

    try {
        const ai = new GoogleGenAI({apiKey: process.env.API_KEY});
        const base64 = originalImage.base64;

        const imagePart = {
            inlineData: {
                data: base64.split(',')[1],
                mimeType: originalImage.file.type
            }
        };
        const textPart = { text: `Analyze the subject in this image. Create a detailed, professional photography background prompt that would complement the subject. The user suggested: "${suggestion}". The background should be realistic and harmonious with the subject's lighting and style. Only return the prompt text.` };

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: { parts: [imagePart, textPart] },
        });

        const promptEl = document.getElementById('background-prompt-input') as HTMLTextAreaElement;
        if(promptEl) {
            promptEl.value = response.text;
            promptEl.dispatchEvent(new Event('input')); 
        }

    } catch (error) {
        console.error("Error generating background prompt:", error);
        alert(`Không thể tạo mô tả: ${(error as Error).message}`);
    } finally {
        isLoading = false;
        spinner.remove();
        const suggestionInputEl = document.getElementById('suggestion-input') as HTMLInputElement;
        suggestionInputEl.value = '';
        generateBtn.classList.remove('hidden');
        updateChangeBackgroundBtnState();
    }
}

/**
 * Handles the main "Change Background" action.
 */
async function handleChangeBackgroundClick() {
     if (isLoading || !originalImage) return;
    isLoading = true;
    updateChangeBackgroundBtnState();
    outputEl.innerHTML = `
        <div class="spinner"></div>
        <p>Đang thay đổi nền... Việc này có thể mất một chút thời gian.</p>
    `;

    try {
        const ai = new GoogleGenAI({apiKey: process.env.API_KEY});
        const promptEl = document.getElementById('background-prompt-input') as HTMLTextAreaElement;
        
        const parts = [];

        // 1. Add Original Image
        parts.push({
            inlineData: {
                data: originalImage.base64.split(',')[1],
                mimeType: originalImage.file.type
            }
        });

        // 2. Add Background (Image or Text)
        let backgroundInstruction = '';
        if (backgroundImage) {
            parts.push({
                inlineData: {
                    data: backgroundImage.base64.split(',')[1],
                    mimeType: backgroundImage.file.type
                }
            });
            backgroundInstruction = "using the second image provided as the new background.";
        } else {
            backgroundInstruction = `placing the subject into a new background described as: "${promptEl.value}".`;
        }

        // 3. Add the main instruction prompt
        const instructionText = `
            **ROLE: Professional Photo Editor.**
            **TASK:** Your task is to expertly replace the background of the first image (the original subject). You must meticulously cut out the subject and place it seamlessly into the new background.
            **INSTRUCTIONS:**
            1.  Identify and isolate the primary subject(s) from the first image.
            2.  Place the isolated subject(s) into the new context ${backgroundInstruction}
            3.  **Crucially, you must create a photorealistic final image.** This means harmonizing lighting, shadows, color grading, and perspective between the subject and the new background. The final result should look like a single, professionally taken photograph, not a composite.
        `;
        parts.push({ text: instructionText });
        lastUsedPrompt = instructionText;

        const modelParams = {
            model: 'gemini-2.5-flash-image-preview',
            contents: { parts: parts },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        };

        const response = await ai.models.generateContent(modelParams);

        const imageUrls: string[] = [];
        let safetyFlagged = false;

        if (response.candidates && response.candidates.length > 0) {
            const candidate = response.candidates[0];
            if (candidate.content && candidate.content.parts && candidate.content.parts.length > 0) {
                const imagePart = candidate.content.parts.find(p => p.inlineData);
                if (imagePart?.inlineData) {
                    imageUrls.push(`data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`);
                }
            } else if (candidate.finishReason === 'SAFETY') {
                safetyFlagged = true;
            }
        }

        if (imageUrls.length > 0) {
            lastGeneratedImageUrl = imageUrls[0];
            imageUrls.forEach(handleSaveImage);
            renderOutput(createImageGrid(imageUrls));
        } else {
             if (safetyFlagged) {
                 renderOutput(`<p class="error">Nội dung đã bị chặn do chính sách an toàn. Vui lòng thử một mô tả hoặc hình ảnh khác.</p>`);
             } else {
                 renderOutput(`<p class="error">Không thể tạo nội dung. Mô hình đã không trả về kết quả hợp lệ hoặc phản hồi trống. Vui lòng thử lại.</p>`);
             }
        }

    } catch (error) {
        console.error(error);
        const errorMessage = `<p class="error">Đã xảy ra lỗi: ${(error as Error).message}</p>`;
        renderOutput(errorMessage);
    } finally {
        isLoading = false;
        updateChangeBackgroundBtnState();
        scrollToResults();
    }
}

/**
 * Handles the main "Restore Photo" action.
 */
async function handleRestoreClick() {
    if (isLoading || !restorationImage) return;
    isLoading = true;
    updateRestoreButtonState();
    outputEl.innerHTML = `
        <div class="spinner"></div>
        <p>Đang phục hồi ảnh... Việc này có thể mất một chút thời gian.</p>
    `;

    try {
        const ai = new GoogleGenAI({apiKey: process.env.API_KEY});

        const colorizeEl = document.getElementById('colorize-checkbox') as HTMLInputElement;
        const faceRotateEl = document.getElementById('face-rotate-checkbox') as HTMLInputElement;
        const outfitSelectEl = document.getElementById('outfit-select') as HTMLSelectElement;
        const customOutfitInputEl = document.getElementById('custom-outfit-input') as HTMLTextAreaElement;

        // Construct the prompt
        let prompt = `
            **ROLE:** You are a world-class expert in digital photo restoration and enhancement.
            **PRIMARY GOAL:** Your primary task is to restore the provided old/damaged photograph. The final result must be sharp, clear, and look as if it were taken with a modern, high-end professional camera. This includes fixing scratches, removing noise, and improving details.
            **ABSOLUTE CONSTRAINT: THE MOST IMPORTANT RULE:** You must preserve the absolute identity and likeness of any person in the photograph. The restored face and features must be 100% identical to the original, only enhanced in clarity and quality. DO NOT alter their identity.
            **MOST REQUIRED:** Retain 100% of the facial features of the person in the original photo. **USE THE UPLOADED IMAGE AS THE MOST ACCURATE REFERENCE FOR THE FACE.** Absolutely do not change the lines, eyes, nose, mouth. Photorealistic studio portrait. Skin shows fine micro-texture and subtle subsurface scattering; eyes tack sharp; hairline blends cleanly with individual strands and natural fly away. Fabric shows authentic weave, seams and natural wrinkles; metals reflect with tiny imperfections. Lighting coherent with scene; natural shadow falloff on cheekbone, jawline and nose. Background has believable micro-details; avoid CGI-clean look. 85mm equivalent, f/2.0 to f/2.8; subject tack sharp, cinematic color grade; confident posture, slight asymmetry.
        `;

        if (colorizeEl.checked) {
            prompt += `\n- **ADDITIONAL TASK: Colorize:** Professionally colorize this photograph. The colors should be realistic, natural, and appropriate for the scene. Skin tones must be lifelike.`;
        }
        if (faceRotateEl.checked) {
            prompt += `\n- **ADDITIONAL TASK: Face Rotation:** The photo contains one person. You must adjust their pose to be front-facing. This rotation must be subtle and natural, while strictly adhering to the **ABSOLUTE CONSTRAINT** of preserving their exact facial identity.`;
        }
        
        let outfitSelection = outfitSelectEl.value;
        if (outfitSelection === 'custom') {
            outfitSelection = customOutfitInputEl.value;
        }

        if (outfitSelection !== 'none' && outfitSelection.trim() !== '') {
            prompt += `\n- **ADDITIONAL TASK: Change Outfit:** Replace the original clothing of the main subject with a '${outfitSelection}'. The new outfit must be seamlessly integrated, with realistic lighting, shadows, and fabric texture that matches the restored image's quality. Ensure the new clothing fits the subject's posture and body shape naturally.`;
        }
        
        prompt += `\n**FINAL OUTPUT FORMAT:** The output must be only the final, restored image.`;

        lastUsedPrompt = prompt;

        const imagePart = {
            inlineData: {
                data: restorationImage.base64.split(',')[1],
                mimeType: restorationImage.file.type
            }
        };

        const modelParams = {
            model: 'gemini-2.5-flash-image-preview',
            contents: { parts: [imagePart, {text: prompt}] },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        };

        const response = await ai.models.generateContent(modelParams);

        const imageUrls: string[] = [];
        let safetyFlagged = false;

        if (response.candidates && response.candidates.length > 0) {
            const candidate = response.candidates[0];
            if (candidate.content && candidate.content.parts && candidate.content.parts.length > 0) {
                const imagePart = candidate.content.parts.find(p => p.inlineData);
                if (imagePart?.inlineData) {
                    imageUrls.push(`data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`);
                }
            } else if (candidate.finishReason === 'SAFETY') {
                safetyFlagged = true;
            }
        }

        if (imageUrls.length > 0) {
            lastGeneratedImageUrl = imageUrls[0];
            imageUrls.forEach(handleSaveImage);
            renderOutput(createImageGrid(imageUrls));
        } else {
             if (safetyFlagged) {
                 renderOutput(`<p class="error">Nội dung đã bị chặn do chính sách an toàn. Vui lòng thử một mô tả hoặc hình ảnh khác.</p>`);
             } else {
                 renderOutput(`<p class="error">Không thể tạo nội dung. Mô hình đã không trả về kết quả hợp lệ hoặc phản hồi trống. Vui lòng thử lại.</p>`);
             }
        }

    } catch (error) {
        console.error(error);
        const errorMessage = `<p class="error">Đã xảy ra lỗi: ${(error as Error).message}</p>`;
        renderOutput(errorMessage);
    } finally {
        isLoading = false;
        updateRestoreButtonState();
        scrollToResults();
    }
}


// --- Inpaint specific functions ---

/**
 * Initializes the canvas elements for the inpaint feature.
 */
function initializeInpaintCanvas(fileData: { base64: string }) {
    const uploadZone = document.getElementById('inpaint-image-upload')!;
    const canvasWrapper = document.getElementById('inpaint-canvas-wrapper')!;
    const tools = document.getElementById('inpaint-tools')!;
    const removeBtn = document.createElement('button');

    uploadZone.classList.add('hidden');
    canvasWrapper.classList.remove('hidden');
    tools.classList.remove('hidden');

    inpaintDisplayCanvas = document.getElementById('inpaint-display-canvas') as HTMLCanvasElement;
    inpaintMaskCanvas = document.createElement('canvas'); // Off-screen canvas
    inpaintOriginalImage = new Image();
    
    inpaintOriginalImage.onload = () => {
        if (!inpaintOriginalImage || !inpaintDisplayCanvas || !inpaintMaskCanvas) return;
        
        const containerWidth = inpaintDisplayCanvas.parentElement!.clientWidth;
        const scale = Math.min(1, containerWidth / inpaintOriginalImage.width);
        const canvasWidth = inpaintOriginalImage.width * scale;
        const canvasHeight = inpaintOriginalImage.height * scale;

        inpaintDisplayCanvas.width = canvasWidth;
        inpaintDisplayCanvas.height = canvasHeight;
        inpaintMaskCanvas.width = canvasWidth;
        inpaintMaskCanvas.height = canvasHeight;
        
        const displayCtx = inpaintDisplayCanvas.getContext('2d')!;
        displayCtx.drawImage(inpaintOriginalImage, 0, 0, canvasWidth, canvasHeight);
        
        inpaintUndoStack = [];
    };
    
    inpaintOriginalImage.src = fileData.base64;

    // Attach canvas drawing listeners
    inpaintDisplayCanvas.addEventListener('mousedown', startInpaintDraw);
    inpaintDisplayCanvas.addEventListener('mousemove', inpaintDraw);
    inpaintDisplayCanvas.addEventListener('mouseup', stopInpaintDraw);
    inpaintDisplayCanvas.addEventListener('mouseleave', stopInpaintDraw);
    inpaintDisplayCanvas.addEventListener('touchstart', startInpaintDraw, { passive: false });
    inpaintDisplayCanvas.addEventListener('touchmove', inpaintDraw, { passive: false });
    inpaintDisplayCanvas.addEventListener('touchend', stopInpaintDraw);


    // Tools Listeners
    const brushSizeSlider = document.getElementById('brush-size') as HTMLInputElement;
    const brushSizeValue = document.getElementById('brush-size-value')!;
    brushSizeSlider.addEventListener('input', (e) => {
        inpaintBrushSize = parseInt((e.target as HTMLInputElement).value, 10);
        brushSizeValue.textContent = String(inpaintBrushSize);
    });
    
    document.getElementById('undo-btn')!.addEventListener('click', handleUndo);
    document.getElementById('clear-mask-btn')!.addEventListener('click', handleClearMask);

    // Add a remove button to go back
    removeBtn.className = 'remove-btn';
    removeBtn.innerHTML = '&times;';
    removeBtn.setAttribute('aria-label', 'Remove image');
    removeBtn.style.position = 'absolute';
    removeBtn.style.top = '5px';
    removeBtn.style.right = '5px';
    canvasWrapper.appendChild(removeBtn);
    removeBtn.addEventListener('click', () => {
        inpaintImage = null;
        inpaintOriginalImage = null;
        inpaintDisplayCanvas = null;
        inpaintMaskCanvas = null;
        inpaintUndoStack = [];
        uploadZone.classList.remove('hidden');
        canvasWrapper.classList.add('hidden');
        tools.classList.add('hidden');
        canvasWrapper.removeChild(removeBtn);
        updateInpaintButtonState();
    });
}

function getEventPosition(event: MouseEvent | TouchEvent) {
    const canvas = inpaintDisplayCanvas!;
    const rect = canvas.getBoundingClientRect();
    if (event instanceof MouseEvent) {
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    } else { // TouchEvent
         return {
            x: event.touches[0].clientX - rect.left,
            y: event.touches[0].clientY - rect.top
        };
    }
}


function startInpaintDraw(event: MouseEvent | TouchEvent) {
    event.preventDefault();
    if (!inpaintMaskCanvas) return;
    inpaintIsDrawing = true;
    
    // Save current state for undo
    const maskCtx = inpaintMaskCanvas.getContext('2d')!;
    inpaintUndoStack.push(maskCtx.getImageData(0, 0, inpaintMaskCanvas.width, inpaintMaskCanvas.height));
    if (inpaintUndoStack.length > 10) inpaintUndoStack.shift(); // Limit undo history
    
    inpaintDraw(event);
}

function stopInpaintDraw() {
    if (inpaintIsDrawing) {
        inpaintIsDrawing = false;
        updateInpaintButtonState();
    }
}

function inpaintDraw(event: MouseEvent | TouchEvent) {
    if (!inpaintIsDrawing || !inpaintDisplayCanvas || !inpaintMaskCanvas) return;
    event.preventDefault();
    
    const maskCtx = inpaintMaskCanvas.getContext('2d')!;
    const pos = getEventPosition(event);

    maskCtx.fillStyle = 'white';
    maskCtx.beginPath();
    maskCtx.arc(pos.x, pos.y, inpaintBrushSize / 2, 0, Math.PI * 2);
    maskCtx.fill();
    
    redrawInpaintDisplay();
}

function redrawInpaintDisplay() {
    if (!inpaintDisplayCanvas || !inpaintOriginalImage || !inpaintMaskCanvas) return;
    const displayCtx = inpaintDisplayCanvas.getContext('2d')!;
    
    displayCtx.clearRect(0, 0, inpaintDisplayCanvas.width, inpaintDisplayCanvas.height);
    displayCtx.drawImage(inpaintOriginalImage, 0, 0, inpaintDisplayCanvas.width, inpaintDisplayCanvas.height);
    
    displayCtx.globalAlpha = 0.5;
    displayCtx.fillStyle = 'red';
    
    // Create a temporary canvas to draw the red overlay
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = inpaintMaskCanvas.width;
    tempCanvas.height = inpaintMaskCanvas.height;
    const tempCtx = tempCanvas.getContext('2d')!;
    
    // Draw mask content
    tempCtx.drawImage(inpaintMaskCanvas, 0, 0);
    // Change composite operation to color the white mask red
    tempCtx.globalCompositeOperation = 'source-in';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Draw the red-colored mask onto the display
    displayCtx.drawImage(tempCanvas, 0, 0);
    
    displayCtx.globalAlpha = 1.0;
}

function handleUndo() {
    if (inpaintUndoStack.length > 0 && inpaintMaskCanvas) {
        const lastState = inpaintUndoStack.pop()!;
        const maskCtx = inpaintMaskCanvas.getContext('2d')!;
        maskCtx.putImageData(lastState, 0, 0);
        redrawInpaintDisplay();
        updateInpaintButtonState();
    }
}

function handleClearMask() {
    if (inpaintMaskCanvas) {
        const maskCtx = inpaintMaskCanvas.getContext('2d')!;
        
        // Save state for undo
        inpaintUndoStack.push(maskCtx.getImageData(0, 0, inpaintMaskCanvas.width, inpaintMaskCanvas.height));
        if (inpaintUndoStack.length > 10) inpaintUndoStack.shift();

        maskCtx.clearRect(0, 0, inpaintMaskCanvas.width, inpaintMaskCanvas.height);
        redrawInpaintDisplay();
        updateInpaintButtonState();
    }
}


async function handleInpaintClick() {
    if (isLoading || !inpaintImage || !inpaintMaskCanvas) return;

    const promptEl = document.getElementById('inpaint-prompt-input') as HTMLTextAreaElement;
    if (!promptEl || promptEl.value.trim() === '') return;

    isLoading = true;
    updateInpaintButtonState();
    outputEl.innerHTML = `
        <div class="spinner"></div>
        <p>Đang chỉnh sửa ảnh... Việc này có thể mất một chút thời gian.</p>
    `;

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        // The mask is already white on a transparent background, which works well.
        const maskBase64 = inpaintMaskCanvas.toDataURL('image/png').split(',')[1];

        const originalImagePart = {
            inlineData: {
                data: inpaintImage.base64.split(',')[1],
                mimeType: inpaintImage.file.type
            }
        };
        const maskImagePart = {
            inlineData: {
                data: maskBase64,
                mimeType: 'image/png'
            }
        };

        const prompt = `
            **ROLE:** AI Photo Inpainting Expert.
            **TASK:** You are given an original image and a mask image. Your task is to modify ONLY the area of the original image that corresponds to the non-transparent parts of the mask image. The rest of the image must remain untouched.
            **MODIFICATION:** The change to make in the masked area is: "${promptEl.value}".
            **IMPORTANT:** The final result must be a seamless, photorealistic blend. The modified area should match the lighting, texture, and style of the surrounding original image.
            **OUTPUT:** Return only the final edited image.
        `;
        const textPart = { text: prompt };
        lastUsedPrompt = prompt;

        const modelParams = {
            model: 'gemini-2.5-flash-image-preview',
            contents: { parts: [originalImagePart, maskImagePart, textPart] },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        };

        const response = await ai.models.generateContent(modelParams);

        const imageUrls: string[] = [];
        let safetyFlagged = false;

        if (response.candidates && response.candidates.length > 0) {
            const candidate = response.candidates[0];
            if (candidate.content && candidate.content.parts && candidate.content.parts.length > 0) {
                const imagePart = candidate.content.parts.find(p => p.inlineData);
                if (imagePart?.inlineData) {
                    imageUrls.push(`data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`);
                }
            } else if (candidate.finishReason === 'SAFETY') {
                safetyFlagged = true;
            }
        }

        if (imageUrls.length > 0) {
            lastGeneratedImageUrl = imageUrls[0];
            imageUrls.forEach(handleSaveImage);
            renderOutput(createImageGrid(imageUrls));
        } else {
             if (safetyFlagged) {
                 renderOutput(`<p class="error">Nội dung đã bị chặn do chính sách an toàn. Vui lòng thử một mô tả hoặc hình ảnh khác.</p>`);
             } else {
                 renderOutput(`<p class="error">Không thể tạo nội dung. Mô hình đã không trả về kết quả hợp lệ hoặc phản hồi trống. Vui lòng thử lại.</p>`);
             }
        }

    } catch (error) {
        console.error(error);
        const errorMessage = `<p class="error">Đã xảy ra lỗi: ${(error as Error).message}</p>`;
        renderOutput(errorMessage);
    } finally {
        isLoading = false;
        updateInpaintButtonState();
        scrollToResults();
    }
}


// --- Functions from the original Generator Tab ---

function handleClearPromptClick() {
    if (promptInputEl) {
        promptInputEl.value = '';
        promptInputEl.dispatchEvent(new Event('input'));
    }
}

function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files) {
        handleFiles(Array.from(target.files));
    }
}

function handleDragOver(event: DragEvent) {
    event.preventDefault();
    (event.currentTarget as HTMLElement).classList.add('active');
}

function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    (event.currentTarget as HTMLElement).classList.remove('active');
}

function handleDrop(event: DragEvent) {
    event.preventDefault();
    (event.currentTarget as HTMLElement).classList.remove('active');
    if (event.dataTransfer?.files) {
        handleFiles(Array.from(event.dataTransfer.files));
    }
}

async function handleFiles(files: File[]) {
    const imageFile = files.find(file => file.type.startsWith('image/'));
    if (imageFile) {
        const base64 = await fileToBase64(imageFile);
        const id = `file-${Date.now()}-${Math.random()}`;
        uploadedFiles = [{ file: imageFile, base64, id }];
    }
    renderImagePreviews();
    updateGenerateButtonState();
}

function renderImagePreviews() {
    if (!imagePreviewEl) return;

    if (uploadedFiles.length > 0) {
        const fileData = uploadedFiles[0];
        imagePreviewEl.innerHTML = `
            <img src="${fileData.base64}" alt="${fileData.file.name}" class="single-preview-img"/>
            <button class="remove-btn" data-id="${fileData.id}" aria-label="Remove image">&times;</button>
        `;
        imagePreviewEl.querySelector('.remove-btn')?.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent parent drop-zone click event
            removeFile(fileData.id);
        });
    } else {
        imagePreviewEl.innerHTML = `<p style="font-size: 14pt">Nhấn(Bấm) ở đây để chọn ảnh</p>`;
    }
}

function removeFile(id: string) {
    uploadedFiles = uploadedFiles.filter(f => f.id !== id);
    renderImagePreviews();
    updateGenerateButtonState();
}

function updateUiForMode() {
    if (activeTab !== 'generate' || !analyzeBtnEl) return;
    
    const aspectRatioWrapperEl = document.getElementById('aspect-ratio-wrapper');
    const keepFaceSectionEl = document.getElementById('keep-face-section');
    
    if (!aspectRatioWrapperEl || !keepFaceSectionEl) return;
    
    const hasImages = uploadedFiles.length > 0;

    keepFaceSectionEl.classList.toggle('hidden', !hasImages);
    aspectRatioWrapperEl.classList.toggle('hidden', hasImages);
    
    analyzeBtnEl.disabled = isLoading;
}

function updateGenerateButtonState() {
    if (activeTab !== 'generate' || !generateBtnEl || !promptInputEl) return;
    generateBtnEl.disabled = promptInputEl.value.trim() === '' || isLoading;
    updateUiForMode();
}

function clearOutput() {
    outputEl.innerHTML = `
        <div class="placeholder">
            <p>Ảnh của bạn sẽ xuất hiện ở đây.</p>
        </div>
    `;
    lastGeneratedImageUrl = null;
    lastUsedPrompt = null;
}

/**
 * Shows a temporary notification on the screen.
 * @param message The message to display.
 */
function showCopyNotification(message: string) {
    const notification = document.createElement('div');
    notification.className = 'copy-notification';
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000); // Notification disappears after 3 seconds
}

/**
 * Creates the HTML for the generated image grid.
 * @param imageUrls An array of image data URLs.
 * @returns The HTML string for the grid.
 */
function createImageGrid(imageUrls: string[]): string {
    return `
        <div class="generated-image-grid">
            ${imageUrls.map((url, index) => `
                <div class="generated-image-container" data-src="${url}" title="Nhấn để phóng to ảnh">
                    <img src="${url}" alt="Generated Image ${index + 1}" class="generated-image">
                    <div class="image-actions">
                        <button class="image-action-btn download-btn" data-src="${url}" aria-label="Tải xuống ảnh ${index + 1}">
                             <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M480-320 280-520l56-58 104 104v-326h80v326l104-104 56 58-200 200ZM240-160q-33 0-56.5-23.5T160-240v-120h80v120h480v-120h80v120q0 33-23.5 56.5T720-160H240Z"/></svg>
                        </button>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Handles saving a generated image to the saved images gallery using IndexedDB.
 * @param imageUrl The data URL of the image to save.
 */
async function handleSaveImage(imageUrl: string) {
    try {
        const id = `saved-${Date.now()}-${Math.random()}`;
        const blob = dataURLtoBlob(imageUrl);

        await addImageToDb({ id, blob });

        // Add to local state using a memory-efficient object URL
        const objectUrl = URL.createObjectURL(blob);
        savedImages.push({ url: objectUrl, id: id });

        renderSavedImages();
    } catch (error) {
        console.error("Failed to save image to IndexedDB:", error);
        // Optionally, inform the user that saving failed.
    }
}

/**
 * Renders the grid of saved images.
 */
function renderSavedImages() {
    if (!savedImagesGridEl) return;

    if (savedImages.length === 0) {
        savedImagesGridEl.innerHTML = `<p class="placeholder-text">Chưa có ảnh nào được lưu.</p>`;
        return;
    }

    savedImagesGridEl.innerHTML = savedImages.map(imgData => `
        <div class="saved-image-container" data-src="${imgData.url}" title="Nhấn để phóng to ảnh">
            <img src="${imgData.url}" alt="Saved image" class="saved-image"/>
            <button class="remove-btn" data-id="${imgData.id}" aria-label="Xóa ảnh đã lưu">&times;</button>
        </div>
    `).join('');

    savedImagesGridEl.querySelectorAll('.remove-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent container's click event from firing
            const idToRemove = (btn as HTMLButtonElement).dataset.id;
            if (idToRemove) {
                handleRemoveSavedImage(idToRemove);
            }
        });
    });

    savedImagesGridEl.querySelectorAll('.saved-image-container').forEach(container => {
        container.addEventListener('click', () => {
            const imageUrl = (container as HTMLElement).dataset.src;
            if (imageUrl) {
                handleImageZoom(imageUrl);
            }
        });
    });
}

/**
 * Removes an image from the saved images gallery and IndexedDB.
 * @param id The ID of the image to remove.
 */
async function handleRemoveSavedImage(id: string) {
    try {
        await deleteImageFromDb(id);

        // Find the image in the local state to revoke its object URL and free up memory
        const imageToRemove = savedImages.find(img => img.id === id);
        if (imageToRemove) {
            URL.revokeObjectURL(imageToRemove.url);
        }

        savedImages = savedImages.filter(img => img.id !== id);
        renderSavedImages();
    } catch (error) {
        console.error("Failed to delete image from IndexedDB:", error);
    }
}


function renderOutput(content: string) {
    // If called with no content, just clear the output to show the placeholder.
    if (!content.trim()) {
        clearOutput();
        return;
    }

    const finalHtml = `
        <div class="output-content">
            ${content}
        </div>
    `;
    outputEl.innerHTML = finalHtml;

    // Attach listeners
    outputEl.querySelectorAll('.download-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent container's click event
            const imageUrl = (e.currentTarget as HTMLButtonElement).dataset.src;
            if (imageUrl) handleImageDownload(imageUrl);
        });
    });
    
    const generatedImgContainerEls = outputEl.querySelectorAll('.generated-image-container');
    generatedImgContainerEls.forEach(containerEl => {
        containerEl.addEventListener('click', () => {
            const imageUrl = (containerEl as HTMLElement).dataset.src;
            if (imageUrl) {
                handleImageZoom(imageUrl);
            }
        });
    });
}

function handleAnalyzeClick() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';

    input.addEventListener('change', async (event) => {
        if (!promptInputEl || !analyzeBtnEl) return;
        const target = event.target as HTMLInputElement;
        const file = target.files?.[0];

        if (!file) return;

        isLoading = true;
        analyzeBtnEl.innerHTML = '<span class="small-spinner"></span> Đang phân tích...';
        updateGenerateButtonState();

        try {
            const ai = new GoogleGenAI({apiKey: process.env.API_KEY});
            const base64 = await fileToBase64(file);

            const imagePart = {
                inlineData: {
                    data: base64.split(',')[1],
                    mimeType: file.type
                }
            };
            const textPart = { text: "Describe this image for a generative AI. Create a detailed prompt that would help generate a similar image with another input image, focusing on subject, style, colors, and composition. Only prompt send back" };

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: { parts: [imagePart, textPart] },
            });

            promptInputEl.value = response.text;
            promptInputEl.dispatchEvent(new Event('input'));

        } catch (error) {
            console.error("Error analyzing image:", error);
            alert(`Không thể phân tích ảnh: ${(error as Error).message}`);
        } finally {
            isLoading = false;
            analyzeBtnEl.innerHTML = 'Phân tích ảnh';
            updateGenerateButtonState();
        }
    });

    input.click();
}

async function handleGenerateClick() {
    if (isLoading || !promptInputEl) return;
    isLoading = true;
    updateGenerateButtonState();
    outputEl.innerHTML = `
        <div class="spinner"></div>
        <p>Đang tạo ảnh... Việc này có thể mất một chút thời gian.</p>
    `;

    try {
        const ai = new GoogleGenAI({apiKey: process.env.API_KEY});

        if (uploadedFiles.length > 0) {
            // EDIT MODE
            const imageParts = uploadedFiles.map(fileData => ({
                inlineData: {
                    data: fileData.base64.split(',')[1],
                    mimeType: fileData.file.type
                }
            }));
            
            let finalPrompt = promptInputEl.value;
            if (keepFaceCheckboxEl && keepFaceCheckboxEl.checked) {
                const facePrompt = "**MOST REQUIRED:** Retain 100% of the facial features of the person in the original photo. **USE THE UPLOADED IMAGE AS THE MOST ACCURATE REFERENCE FOR THE FACE.** Absolutely do not change the lines, eyes, nose, mouth. Photorealistic studio portrait. Skin shows fine micro-texture and subtle subsurface scattering; eyes tack sharp; hairline blends cleanly with individual strands and natural fly away. Fabric shows authentic weave, seams and natural wrinkles; metals reflect with tiny imperfections. Lighting coherent with scene; natural shadow falloff on cheekbone, jawline and nose. Background has believable micro-details; avoid CGI-clean look. 85mm equivalent, f/2.0–f/2.8; subject tack sharp, cinematic color grade; confident posture, slight asymmetry.";
                finalPrompt = `${facePrompt} ${promptInputEl.value}`;
            }
            const textPart = { text: finalPrompt };
            lastUsedPrompt = finalPrompt;
            
            const modelParams = {
                model: 'gemini-2.5-flash-image-preview',
                contents: { parts: [...imageParts, textPart] },
                config: {
                    responseModalities: [Modality.IMAGE, Modality.TEXT],
                },
            };

            const response = await ai.models.generateContent(modelParams);

            const imageUrls: string[] = [];
            let safetyFlagged = false;

            if (response.candidates && response.candidates.length > 0) {
                const candidate = response.candidates[0];
                if (candidate.content && candidate.content.parts && candidate.content.parts.length > 0) {
                    const imagePart = candidate.content.parts.find(p => p.inlineData);
                    if (imagePart?.inlineData) {
                        imageUrls.push(`data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`);
                    }
                } else if (candidate.finishReason === 'SAFETY') {
                    safetyFlagged = true;
                }
            }

            if (imageUrls.length > 0) {
                lastGeneratedImageUrl = imageUrls[0];
                imageUrls.forEach(handleSaveImage);
                renderOutput(createImageGrid(imageUrls));
            } else {
                 if (safetyFlagged) {
                     renderOutput(`<p class="error">Nội dung đã bị chặn do chính sách an toàn. Vui lòng thử một mô tả hoặc hình ảnh khác.</p>`);
                 } else {
                     renderOutput(`<p class="error">Không thể tạo nội dung. Mô hình đã không trả về kết quả hợp lệ hoặc phản hồi trống. Vui lòng thử lại.</p>`);
                 }
            }

        } else {
            // GENERATE MODE
             if(!aspectRatioEl) throw new Error("Aspect ratio element not found");
             lastUsedPrompt = promptInputEl.value;
            const response = await ai.models.generateImages({
                model: 'imagen-4.0-generate-001',
                prompt: promptInputEl.value,
                config: {
                  numberOfImages: 1,
                  outputMimeType: 'image/jpeg',
                  aspectRatio: aspectRatioEl.value as "1:1" | "3:4" | "4:3" | "9:16" | "16:9",
                },
            });
            
            if (response.generatedImages && response.generatedImages.length > 0) {
                const imageUrls = response.generatedImages.map(genImage => `data:image/jpeg;base64,${genImage.image.imageBytes}`);
                lastGeneratedImageUrl = imageUrls[0];
                imageUrls.forEach(handleSaveImage);
                renderOutput(createImageGrid(imageUrls));
            } else {
                renderOutput(`<p class="error">Không thể tạo ảnh từ mô tả. Điều này có thể do chính sách an toàn.</p>`);
            }
        }

    } catch (error) {
        console.error(error);
        const errorMessage = `<p class="error">Đã xảy ra lỗi: ${(error as Error).message}</p>`;
        renderOutput(errorMessage);
    } finally {
        isLoading = false;
        if(activeTab === 'generate') updateGenerateButtonState();
        scrollToResults();
    }
}

function handleImageZoom(imageUrl: string) {
    if (imageUrl) {
        zoomedImgEl.src = imageUrl;
        zoomModalEl.style.display = 'flex';
    }
}

function closeZoomModal() {
    zoomModalEl.style.display = 'none';
}

function handleImageDownload(imageUrl: string) {
    if (imageUrl) {
        const a = document.createElement('a');
        a.href = imageUrl;
        a.download = `generated-image-${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = error => reject(error);
    });
}

App();