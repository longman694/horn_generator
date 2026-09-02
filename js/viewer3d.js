/**
 * Horn 3D Viewer Manager using Three.js
 * Provides interactive 3D rendering with shading, wireframe, cutaway cross-section, and orbit controls.
 */

class Horn3DViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found.`);
            return;
        }

        this.wireframeMode = false;
        this.cutawayMode = false;
        this.autoRotate = false;
        this.wallThickness = 2.0;

        this.initScene();
        this.animate();
        this.handleResize();
    }

    initScene() {
        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 600;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f172a); // Slate dark background

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 5000);
        this.camera.position.set(200, 150, 250);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.localClippingEnabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Orbit Controls
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
        }

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const mainLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
        mainLight.position.set(300, 400, 200);
        mainLight.castShadow = true;
        this.scene.add(mainLight);

        const fillLight = new THREE.DirectionalLight(0xf43f5e, 0.6);
        fillLight.position.set(-200, -200, -100);
        this.scene.add(fillLight);

        const topLight = new THREE.DirectionalLight(0xffffff, 0.8);
        topLight.position.set(0, 500, 0);
        this.scene.add(topLight);

        // Grid & Axis Helpers
        this.gridHelper = new THREE.GridHelper(400, 40, 0x334155, 0x1e293b);
        this.gridHelper.position.y = -50;
        this.scene.add(this.gridHelper);

        this.axesHelper = new THREE.AxesHelper(50);
        this.scene.add(this.axesHelper);

        // Clipping plane for cutaway half-section
        this.clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);

        // Materials
        this.solidMaterial = new THREE.MeshStandardMaterial({
            color: 0x0284c7, // Sky blue cyan
            metalness: 0.3,
            roughness: 0.25,
            side: THREE.DoubleSide,
            clippingPlanes: this.cutawayMode ? [this.clipPlane] : []
        });

        this.wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            wireframe: true
        });

        // Group holding the horn mesh
        this.hornGroup = new THREE.Group();
        this.scene.add(this.hornGroup);

        window.addEventListener('resize', () => this.handleResize());
    }

    updateMesh(points, isHCD = false, wallThickness = 2.0) {
        this.wallThickness = wallThickness;
        if (!points || points.length === 0) return;

        // Clear previous meshes
        while (this.hornGroup.children.length > 0) {
            const obj = this.hornGroup.children.pop();
            if (obj.geometry) obj.geometry.dispose();
        }

        // Build raw mesh geometry using Exporters helper
        const { vertices, triangles } = HornExporters.buildHornMeshGeometry(points, isHCD, wallThickness, 96);

        const geometry = new THREE.BufferGeometry();

        const flatVertices = [];
        const flatIndices = [];

        for (let i = 0; i < vertices.length; i++) {
            flatVertices.push(vertices[i]);
        }

        for (let t = 0; t < triangles.length; t++) {
            flatIndices.push(triangles[t][0], triangles[t][1], triangles[t][2]);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(flatVertices, 3));
        geometry.setIndex(flatIndices);
        geometry.computeVertexNormals();

        // Update Material Clipping
        this.solidMaterial.clippingPlanes = this.cutawayMode ? [this.clipPlane] : [];

        // Create Mesh
        const meshMat = this.wireframeMode ? this.wireframeMaterial : this.solidMaterial;
        this.mainMesh = new THREE.Mesh(geometry, meshMat);
        this.hornGroup.add(this.mainMesh);

        // Optional wireframe outline overlay when solid
        if (!this.wireframeMode) {
            const wireGeo = new THREE.WireframeGeometry(geometry);
            const wireMat = new THREE.LineBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.25 });
            const wireMesh = new THREE.LineSegments(wireGeo, wireMat);
            this.hornGroup.add(wireMesh);
        }

        // Center object in scene
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox;
        const center = new THREE.Vector3();
        bbox.getCenter(center);
        this.hornGroup.position.set(-center.x, -center.y, -center.z);

        if (this.controls) {
            this.controls.target.set(0, 0, 0);
        }
    }

    setWireframe(enabled) {
        this.wireframeMode = enabled;
        if (this.mainMesh) {
            this.mainMesh.material = enabled ? this.wireframeMaterial : this.solidMaterial;
        }
    }

    setCutaway(enabled) {
        this.cutawayMode = enabled;
        this.solidMaterial.clippingPlanes = enabled ? [this.clipPlane] : [];
        this.solidMaterial.needsUpdate = true;
    }

    setAutoRotate(enabled) {
        this.autoRotate = enabled;
        if (this.controls) {
            this.controls.autoRotate = enabled;
            this.controls.autoRotateSpeed = 2.0;
        }
    }

    resetCamera() {
        this.camera.position.set(200, 150, 250);
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    handleResize() {
        if (!this.container) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        if (width > 0 && height > 0) {
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        this.renderer.render(this.scene, this.camera);
    }
}

window.Horn3DViewer = Horn3DViewer;
