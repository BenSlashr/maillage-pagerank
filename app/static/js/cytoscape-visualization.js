/**
 * Visualisation de graphe avec Cytoscape.js
 * Une alternative plus performante pour les grands graphes
 */

// Configuration globale
const cyConfig = {
    // Couleurs pour les différents types de nœuds et d'arêtes
    colors: {
        nodes: {
            default: '#6c757d',
            category: '#007bff',
            product: '#28a745',
            blog: '#fd7e14',
            homepage: '#dc3545'
        },
        edges: {
            existing: '#6c757d',
            suggested: '#28a745',
            frequent: '#cccccc' // Couleur plus claire pour les liens fréquents
        },
        depth: [
            '#3366cc', // Niveau 0 (racine)
            '#dc3912', // Niveau 1
            '#ff9900', // Niveau 2
            '#109618', // Niveau 3
            '#990099', // Niveau 4
            '#0099c6', // Niveau 5
            '#dd4477', // Niveau 6
            '#66aa00', // Niveau 7
            '#b82e2e', // Niveau 8
            '#316395'  // Niveau 9+
        ]
    },
    // Taille des nœuds
    nodeSize: {
        min: 5,
        max: 30,
        factor: 1000
    },
    // Mise en page
    layout: {
        name: 'cose',
        idealEdgeLength: 100,
        nodeOverlap: 20,
        refresh: 20,
        fit: true,
        padding: 30,
        randomize: false,
        componentSpacing: 100,
        nodeRepulsion: 400000,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0
    }
};

// Variables globales
let cyCurrentNetwork = null;
let cyOptimizedNetwork = null;
let graphData = null;
let pagerankData = null;
let visualizationConfig = null;
let depthData = null;

// Configuration pour le filtrage des nœuds
let nodeDisplayConfig = {
    maxNodes: 100,  // Nombre maximum de nœuds à afficher par défaut
    filterBy: 'pagerank',  // Critère de filtrage ('pagerank', 'links', 'all')
    maxDepth: null,  // Profondeur maximale à afficher (null = toutes)
    hideFrequentLinks: true,  // Masquer les liens fréquents (menu/footer)
    colorByDepth: true  // Colorer les nœuds par profondeur
};

/**
 * Initialise la visualisation Cytoscape
 * @param {Object} data - Données du graphe
 */
function initCytoscapeVisualization(data) {
    console.log('Initialisation de la visualisation Cytoscape avec les données:', data);
    
    // Vérifier que les données sont valides
    if (!data || !data.graph) {
        console.error('Données de graphe manquantes');
        showError('Les données de graphe sont manquantes ou invalides.');
        return;
    }
    
    graphData = data.graph;
    pagerankData = data.pagerank;
    depthData = data.depth_data || null;
    visualizationConfig = data.config || {
        use_weighted_pagerank: false,
        content_links_only: false,
        alpha: 0.5,
        beta: 0.5,
        filter_frequent_links: true, // Activé par défaut
        frequent_links_threshold: 80.0,
        root_url: null
    };
    
    // Forcer l'activation des options par défaut
    visualizationConfig.filter_frequent_links = true;
    nodeDisplayConfig.colorByDepth = true;
    nodeDisplayConfig.hideFrequentLinks = true;
    
    // Configurer les liens avec l'ID du job
    const backToResultsLink = document.getElementById('back-to-results');
    if (backToResultsLink) {
        const jobId = window.location.pathname.split('/').pop();
        backToResultsLink.href = `/results/${jobId}`;
    }
    
    // Initialiser les contrôles de filtrage par profondeur si les données sont disponibles
    if (depthData) {
        initDepthControls(depthData.max_depth);
    }
    
    // D'abord configurer les options par défaut
    setupDefaultOptions();
    
    // Ensuite mettre à jour les autres contrôles
    updateControlsFromConfig();
    
    // Ajouter les événements pour les options avancées
    setupAdvancedOptions();
    
    // Vérifier que les donnœuds sont disponibles
    if (!graphData.nodes || graphData.nodes.length === 0) {
        console.error('Aucun nœud dans les données du graphe');
        showError('Aucune page HTML n\'a été trouvée dans les données. Vérifiez que vos URLs correspondent à des pages HTML.');
        return;
    }
    
    // Vérifier que les arêtes sont disponibles
    if (!graphData.edges) {
        console.error('Structure d\'arêtes manquante');
        graphData.edges = { current: [], suggested: [] };
    }
    
    if (!graphData.edges.current) {
        console.warn('Aucune arête existante dans les données');
        graphData.edges.current = [];
    }
    
    if (!graphData.edges.suggested) {
        console.warn('Aucune arête suggérée dans les données');
        graphData.edges.suggested = [];
    }
    
    // Initialiser les réseaux
    initCytoscapeNetwork('cy-current-graph', 'current');
    initCytoscapeNetwork('cy-optimized-graph', 'optimized');
    
    // Afficher les statistiques
    displayNetworkStats();
    
    // Afficher les statistiques de profondeur si disponibles
    if (depthData) {
        displayDepthStats();
    }
    
    // Masquer explicitement tous les spinners après l'initialisation complète
    hideAllSpinners();
}

/**
 * Masque tous les spinners de chargement
 */
function hideAllSpinners() {
    // Masquer les spinners principaux
    const spinners = ['cy-current-spinner', 'cy-optimized-spinner'];
    spinners.forEach(spinnerId => {
        const spinner = document.getElementById(spinnerId);
        if (spinner) {
            spinner.style.display = 'none';
        }
    });
}

/**
 * Affiche un message d'erreur dans l'interface
 * @param {string} message - Message d'erreur
 */
function showError(message) {
    // Afficher l'erreur dans les conteneurs de graphe
    const containers = ['cy-current-graph', 'cy-optimized-graph'];
    
    containers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            // Masquer le spinner
            const spinner = document.getElementById(`${containerId.replace('graph', 'spinner')}`);
            if (spinner) {
                spinner.style.display = 'none';
            }
            
            // Afficher le message d'erreur
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger m-3';
            errorDiv.innerHTML = `<strong>Erreur:</strong> ${message}`;
            container.appendChild(errorDiv);
        }
    });
    
    // Afficher l'erreur dans les statistiques
    const statsContainer = document.getElementById('network-stats');
    if (statsContainer) {
        statsContainer.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Statistiques du réseau</h5>
                </div>
                <div class="card-body">
                    <div class="alert alert-danger">
                        <strong>Erreur:</strong> ${message}
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * Initialise un réseau Cytoscape
 * @param {string} containerId - ID du conteneur
 * @param {string} mode - Mode de visualisation ('current' ou 'optimized')
 */
function initCytoscapeNetwork(containerId, mode) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Préparer les éléments
    const elements = prepareCytoscapeElements(mode);
    
    // Créer l'instance Cytoscape
    const cy = cytoscape({
        container: container,
        elements: elements,
        style: [
            // Style des nœuds
            {
                selector: 'node',
                style: {
                    'background-color': 'data(color)',
                    'label': 'data(label)',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'font-size': '10px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-outline-width': 2,
                    'text-outline-color': '#fff',
                    'color': '#000'
                }
            },
            // Style des arêtes
            {
                selector: 'edge',
                style: {
                    'width': 1,
                    'line-color': 'data(color)',
                    'target-arrow-color': 'data(color)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            },
            // Style des arêtes suggérées
            {
                selector: 'edge[suggested]',
                style: {
                    'line-style': 'dashed',
                    'line-dash-pattern': [6, 3]
                }
            }
        ],
        layout: cyConfig.layout
    });
    
    // Ajouter des événements
    cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        const event = evt.originalEvent;
        console.log('Noeud cliqué:', node.id(), 'Position:', event.pageX, event.pageY);
        showNodeDetails(node.data(), event);
    });
    
    // Ajouter un événement de clic direct sur le conteneur pour débogage
    container.addEventListener('click', function(event) {
        console.log('Clic sur le conteneur:', event.pageX, event.pageY);
    });
    
    // Masquer la bulle de détails lors du déplacement du graphe
    cy.on('pan', function() {
        hideNodeTooltip();
    });
    
    cy.on('zoom', function() {
        hideNodeTooltip();
    });
    
    // Stocker l'instance
    if (mode === 'current') {
        cyCurrentNetwork = cy;
    } else {
        cyOptimizedNetwork = cy;
    }
    
    // Ajouter des contrôles de zoom
    addZoomControls(containerId, cy);
    
    return cy;
}

/**
 * Prépare les éléments pour Cytoscape
 * @param {string} mode - Mode de visualisation ('current' ou 'optimized')
 * @returns {Array} - Éléments pour Cytoscape
 */
function prepareCytoscapeElements(mode) {
    if (!graphData || !graphData.nodes) return [];
    
    // Filtrer les nœuds par PageRank et profondeur
    let nodes = [...graphData.nodes];
    
    // Filtrer par profondeur si configuré
    if (nodeDisplayConfig.maxDepth !== null && depthData) {
        nodes = nodes.filter(node => {
            const depth = node.metrics && node.metrics.depth !== undefined ? node.metrics.depth : Infinity;
            return depth <= nodeDisplayConfig.maxDepth;
        });
    }
    
    // Filtrer par PageRank si nécessaire
    if (nodeDisplayConfig.maxNodes > 0 && nodeDisplayConfig.maxNodes < nodes.length) {
        // Ajouter le PageRank à chaque nœud
        nodes = nodes.map(node => {
            let pagerank = 0;
            if (pagerankData && pagerankData.current && node.id in pagerankData.current) {
                pagerank = pagerankData.current[node.id].pagerank;
            }
            return { ...node, pagerank };
        });
        
        // Trier et filtrer
        nodes = nodes.sort((a, b) => b.pagerank - a.pagerank).slice(0, nodeDisplayConfig.maxNodes);
    }
    
    // Créer un Set des IDs de nœuds pour un filtrage efficace
    const nodeIds = new Set(nodes.map(node => node.id));
    
    // Préparer les nœuds pour Cytoscape
    const cyNodes = nodes.map(node => {
        // Déterminer la taille du nœud en fonction du PageRank
        let size = cyConfig.nodeSize.min;
        if (pagerankData && pagerankData.current && node.id in pagerankData.current) {
            const pagerank = pagerankData.current[node.id].pagerank;
            size = Math.max(
                cyConfig.nodeSize.min,
                Math.min(cyConfig.nodeSize.max, pagerank * cyConfig.nodeSize.factor)
            );
        }
        
        // Déterminer la couleur en fonction du type ou de la profondeur
        let color = cyConfig.colors.nodes.default;
        
        if (nodeDisplayConfig.colorByDepth && node.metrics && node.metrics.depth !== undefined) {
            // Colorer par profondeur
            const depth = node.metrics.depth;
            const colorIndex = Math.min(depth, cyConfig.colors.depth.length - 1);
            color = cyConfig.colors.depth[colorIndex];
        } else if (node.type in cyConfig.colors.nodes) {
            // Colorer par type
            color = cyConfig.colors.nodes[node.type];
        }
        
        return {
            data: {
                id: node.id,
                label: node.label || node.id.split('/').pop(),
                color: color,
                size: size,
                type: node.type,
                originalData: node
            }
        };
    });
    
    // Préparer les arêtes pour Cytoscape
    let cyEdges = [];
    
    // Ajouter les liens existants (uniquement entre nœuds visibles)
    if (graphData.edges.current) {
        const currentEdges = graphData.edges.current
            .filter(edge => {
                // Filtrer par visibilité des nœuds
                const nodesVisible = nodeIds.has(edge.source) && nodeIds.has(edge.target);
                
                // Filtrer les liens fréquents si demandé
                const frequentLinkVisible = !nodeDisplayConfig.hideFrequentLinks || !edge.frequent;
                
                return nodesVisible && frequentLinkVisible;
            })
            .map(edge => ({
                data: {
                    id: `e-${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    color: edge.frequent ? cyConfig.colors.edges.frequent : cyConfig.colors.edges.existing,
                    frequent: edge.frequent || false,
                    originalData: edge
                }
            }));
        
        cyEdges = cyEdges.concat(currentEdges);
    }
    
    // Ajouter les liens suggérés si on est en mode optimisé
    if (mode === 'optimized' && graphData.edges.suggested) {
        const suggestedEdges = graphData.edges.suggested
            .filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target))
            .map(edge => ({
                data: {
                    id: `s-${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    color: cyConfig.colors.edges.suggested,
                    suggested: true,
                    anchor: edge.anchor || '',
                    originalData: edge
                }
            }));
        
        cyEdges = cyEdges.concat(suggestedEdges);
    }
    
    console.log(`Cytoscape: ${cyNodes.length} nœuds et ${cyEdges.length} liens préparés pour le mode ${mode}`);
    return [...cyNodes, ...cyEdges];
}

/**
 * Ajoute des contrôles de zoom à un graphe Cytoscape
 * @param {string} containerId - ID du conteneur
 * @param {Object} cy - Instance Cytoscape
 */
function addZoomControls(containerId, cy) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Créer les boutons
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'cy-controls';
    controlsDiv.innerHTML = `
        <button class="cy-zoom-in" title="Zoom in">+</button>
        <button class="cy-zoom-out" title="Zoom out">-</button>
        <button class="cy-zoom-reset" title="Reset zoom">⟲</button>
    `;
    container.appendChild(controlsDiv);
    
    // Ajouter les événements
    container.querySelector('.cy-zoom-in').addEventListener('click', () => {
        cy.zoom(cy.zoom() * 1.2);
        cy.center();
    });
    
    container.querySelector('.cy-zoom-out').addEventListener('click', () => {
        cy.zoom(cy.zoom() / 1.2);
        cy.center();
    });
    
    container.querySelector('.cy-zoom-reset').addEventListener('click', () => {
        cy.fit();
    });
}

/**
 * Affiche les détails d'un nœud dans une bulle à côté de la souris
 * @param {Object} nodeData - Données du nœud
 * @param {Object} event - Événement de clic (pour la position)
 */
function showNodeDetails(nodeData, event) {
    // Masquer d'abord toute bulle existante
    hideNodeTooltip();
    
    const tooltip = document.getElementById('node-tooltip');
    if (!tooltip) {
        console.error('Conteneur de bulle introuvable!');
        return;
    }
    
    console.log('Affichage des détails pour:', nodeData.id);
    
    // Préparer les détails de PageRank
    let pagerankDetails = '';
    if (pagerankData && pagerankData.current && nodeData.id in pagerankData.current) {
        const current = pagerankData.current[nodeData.id];
        pagerankDetails += `
            <div class="mb-2">
                <h6 class="mb-1">PageRank</h6>
                <div><strong>Score:</strong> ${current.pagerank.toFixed(6)}</div>
                <div><strong>Rang:</strong> ${current.rank} sur ${Object.keys(pagerankData.current).length}</div>
            </div>
        `;
        
        if (pagerankData.optimized && nodeData.id in pagerankData.optimized) {
            const optimized = pagerankData.optimized[nodeData.id];
            const improvement = pagerankData.improvement[nodeData.id];
            
            const improvementClass = improvement.percentage > 0 ? 'text-success' : 
                                   improvement.percentage < 0 ? 'text-danger' : '';
            
            const rankChangeClass = improvement.rank_change > 0 ? 'text-success' : 
                                  improvement.rank_change < 0 ? 'text-danger' : '';
            
            const rankChangeText = improvement.rank_change > 0 ? `+${improvement.rank_change}` : 
                                 improvement.rank_change < 0 ? `${improvement.rank_change}` : '0';
            
            pagerankDetails += `
                <div class="mt-2">
                    <h6 class="mb-1">Optimisation</h6>
                    <div><strong>Amélioration:</strong> <span class="${improvementClass}">${improvement.percentage.toFixed(2)}%</span></div>
                    <div><strong>Changement de rang:</strong> <span class="${rankChangeClass}">${rankChangeText}</span></div>
                </div>
            `;
        }
    }
    
    // Afficher les informations de profondeur si disponibles
    let depthInfo = '';
    
    // Récupérer la profondeur depuis les différentes structures possibles
    let depth;
    
    // Cas 1: Données directement dans nodeData.data.originalData.metrics (structure Cytoscape)
    if (nodeData.data && nodeData.data.originalData && nodeData.data.originalData.metrics && nodeData.data.originalData.metrics.depth !== undefined) {
        depth = nodeData.data.originalData.metrics.depth;
    }
    // Cas 2: Données dans nodeData.data.metrics
    else if (nodeData.data && nodeData.data.metrics && nodeData.data.metrics.depth !== undefined) {
        depth = nodeData.data.metrics.depth;
    }
    // Cas 3: Données directement dans nodeData.metrics
    else if (nodeData.metrics && nodeData.metrics.depth !== undefined) {
        depth = nodeData.metrics.depth;
    }
    
    // Afficher la profondeur si elle a été trouvée
    if (depth !== undefined) {
        depthInfo = `<div class="mt-2"><strong>Profondeur:</strong> ${depth}</div>`;
    }
    
    // Extraire le nom de la page à partir de l'URL
    const pageName = nodeData.id.split('/').pop() || nodeData.id;
    
    // Afficher les détails dans la bulle
    tooltip.innerHTML = `
        <h5>${pageName}</h5>
        <div class="url">${nodeData.id}</div>
        <div class="mt-2">
            <span class="badge bg-secondary">${nodeData.type || 'Page'}</span>
        </div>
        ${depthInfo}
        ${pagerankDetails}
        <div class="mt-3">
            <button class="btn btn-sm btn-primary" onclick="focusOnNode('${nodeData.id}'); hideNodeTooltip();">
                Centrer sur cette page
            </button>
            <button class="btn btn-sm btn-outline-secondary ms-2" onclick="hideNodeTooltip();">
                Fermer
            </button>
        </div>
    `;
    
    // Positionner directement la bulle
    const x = event.pageX + 20;
    const y = event.pageY - 10;
    
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
    tooltip.style.display = 'block';
    
    // Ajouter un événement pour fermer la bulle en cliquant ailleurs
    setTimeout(() => {
        document.addEventListener('click', hideNodeTooltipOnClickOutside);
    }, 100);
}

/**
 * Masque la bulle de détails
 */
function hideNodeTooltip() {
    const tooltip = document.getElementById('node-tooltip');
    if (tooltip) {
        tooltip.style.display = 'none';
    }
    
    // Retirer l'événement de fermeture
    document.removeEventListener('click', hideNodeTooltipOnClickOutside);
}

/**
 * Ferme la bulle de détails si on clique en dehors
 * @param {Event} event - Événement de clic
 */
function hideNodeTooltipOnClickOutside(event) {
    const tooltip = document.getElementById('node-tooltip');
    if (tooltip && !tooltip.contains(event.target)) {
        hideNodeTooltip();
    }
}

/**
 * Centre la vue sur un nœud spécifique
 * @param {string} nodeId - ID du nœud
 */
function focusOnNode(nodeId) {
    const activeTab = document.querySelector('.tab-pane.active');
    if (!activeTab) return;
    
    let cy = null;
    if (activeTab.id === 'cy-current-network') {
        cy = cyCurrentNetwork;
    } else if (activeTab.id === 'cy-optimized-network') {
        cy = cyOptimizedNetwork;
    }
    
    if (cy) {
        const node = cy.getElementById(nodeId);
        if (node.length > 0) {
            cy.fit(node, 50);
            cy.center(node);
        }
    }
}

/**
 * Met à jour le nombre maximum de nœuds à afficher
 * @param {number} newMaxNodes - Nouveau nombre maximum de nœuds
 */
function updateCyMaxNodes(newMaxNodes) {
    nodeDisplayConfig.maxNodes = newMaxNodes;
    
    // Mettre à jour les visualisations
    if (cyCurrentNetwork) {
        const elements = prepareCytoscapeElements('current');
        cyCurrentNetwork.elements().remove();
        cyCurrentNetwork.add(elements);
        cyCurrentNetwork.layout(cyConfig.layout).run();
    }
    
    if (cyOptimizedNetwork) {
        const elements = prepareCytoscapeElements('optimized');
        cyOptimizedNetwork.elements().remove();
        cyOptimizedNetwork.add(elements);
        cyOptimizedNetwork.layout(cyConfig.layout).run();
    }
    
    // Mettre à jour les statistiques
    displayNetworkStats();
}

/**
 * Affiche les statistiques du réseau
 */
function displayNetworkStats() {
    const statsContainer = document.getElementById('network-stats');
    if (!statsContainer || !graphData) return;
    
    const totalNodes = graphData.nodes.length;
    const displayedNodes = nodeDisplayConfig.maxNodes > 0 ? 
                          Math.min(nodeDisplayConfig.maxNodes, totalNodes) : 
                          totalNodes;
    
    const totalCurrentEdges = graphData.edges.current ? graphData.edges.current.length : 0;
    const totalSuggestedEdges = graphData.edges.suggested ? graphData.edges.suggested.length : 0;
    
    // Information sur le type de PageRank utilisé
    const pagerankType = visualizationConfig?.use_weighted_pagerank ? 'pondéré' : 'standard';
    const contentLinksOnly = visualizationConfig?.content_links_only ? 'Oui' : 'Non';
    
    // Informations supplémentaires pour le PageRank pondéré
    let pagerankInfo = '';
    if (visualizationConfig?.use_weighted_pagerank) {
        pagerankInfo = `
            <div class="alert alert-info mt-3">
                <h6 class="alert-heading">Configuration du PageRank pondéré</h6>
                <p class="mb-1"><strong>Coefficient sémantique (α):</strong> ${visualizationConfig.alpha.toFixed(1)}</p>
                <p class="mb-1"><strong>Coefficient position (β):</strong> ${visualizationConfig.beta.toFixed(1)}</p>
                <p class="mb-0"><small>Un coefficient plus élevé donne plus d'importance au facteur correspondant</small></p>
            </div>
        `;
    }
    
    statsContainer.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Statistiques du réseau</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Structure du site</h6>
                        <p><strong>Pages totales:</strong> ${totalNodes}</p>
                        <p><strong>Pages affichées:</strong> ${displayedNodes} (${Math.round(displayedNodes/totalNodes*100)}%)</p>
                        <p><strong>Liens actuels:</strong> ${totalCurrentEdges}</p>
                        <p><strong>Liens suggérés:</strong> ${totalSuggestedEdges}</p>
                        <p><strong>Densité du réseau:</strong> ${(totalCurrentEdges / (totalNodes * (totalNodes - 1))).toFixed(5)}</p>
                    </div>
                    <div class="col-md-6">
                        <h6>Configuration PageRank</h6>
                        <p><strong>Type de PageRank:</strong> ${pagerankType}</p>
                        <p><strong>Liens de contenu uniquement:</strong> ${contentLinksOnly}</p>
                        ${pagerankInfo}
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Change le layout du graphe
 * @param {string} layoutName - Nom du layout
 */
function changeLayout(layoutName) {
    let layoutConfig = { ...cyConfig.layout, name: layoutName };
    
    // Ajuster les paramètres en fonction du layout
    switch (layoutName) {
        case 'cose':
            // Configuration par défaut
            break;
        case 'concentric':
            layoutConfig = {
                name: 'concentric',
                concentric: function(node) {
                    return node.data('size');
                },
                levelWidth: function() {
                    return 2;
                },
                minNodeSpacing: 50
            };
            break;
        case 'breadthfirst':
            layoutConfig = {
                name: 'breadthfirst',
                directed: true,
                spacingFactor: 1.5
            };
            break;
        case 'grid':
            layoutConfig = {
                name: 'grid',
                avoidOverlap: true,
                spacingFactor: 1.5
            };
            break;
        case 'circle':
            layoutConfig = {
                name: 'circle',
                avoidOverlap: true,
                spacingFactor: 1.5
            };
            break;
        case 'depth':
            // Layout hiérarchique par niveau de profondeur
            layoutConfig = {
                name: 'breadthfirst',
                directed: true,
                spacingFactor: 1.5,
                roots: depthData ? findRootNodes() : undefined
            };
            break;
    }
    
    // Appliquer le nouveau layout
    const activeTab = document.querySelector('.tab-pane.active');
    if (!activeTab) return;
    
    let cy = null;
    if (activeTab.id === 'cy-current-network') {
        cy = cyCurrentNetwork;
    } else if (activeTab.id === 'cy-optimized-network') {
        cy = cyOptimizedNetwork;
    }
    
    if (cy) {
        cy.layout(layoutConfig).run();
    }
}

/**
 * Trouve les nœuds racines (profondeur 0) pour le layout hiérarchique
 */
function findRootNodes() {
    if (!depthData || !graphData) return [];
    
    const rootNodes = [];
    for (const node of graphData.nodes) {
        if (node.metrics && node.metrics.depth === 0) {
            rootNodes.push(node.id);
        }
    }
    
    return rootNodes;
}

/**
 * Initialise les contrôles de filtrage par profondeur
 */
function initDepthControls(maxDepth) {
    const depthControlsDiv = document.getElementById('depth-controls');
    if (!depthControlsDiv) return;
    
    // Créer le slider de profondeur
    depthControlsDiv.innerHTML = `
        <div class="card">
            <div class="card-header py-2">
                <h5 class="mb-0">Filtrage par profondeur</h5>
            </div>
            <div class="card-body py-2">
                <div class="mb-2">
                    <label for="depth-slider" class="form-label mb-0">Profondeur maximale: <span id="depth-value">Toutes</span></label>
                    <input type="range" class="form-range" id="depth-slider" min="0" max="${maxDepth}" step="1" value="${maxDepth}">
                </div>
            </div>
        </div>
    `;
    
    // Ajouter les événements
    const depthSlider = document.getElementById('depth-slider');
    const depthValue = document.getElementById('depth-value');
    
    if (depthSlider) {
        depthSlider.addEventListener('input', function() {
            const value = parseInt(this.value);
            if (value === maxDepth) {
                depthValue.textContent = 'Toutes';
                nodeDisplayConfig.maxDepth = null;
            } else {
                depthValue.textContent = value;
                nodeDisplayConfig.maxDepth = value;
            }
            updateVisualization();
        });
    }
}

/**
 * Met à jour la visualisation en fonction des filtres
 */
function updateVisualization() {
    // Mettre à jour les visualisations
    if (cyCurrentNetwork) {
        const elements = prepareCytoscapeElements('current');
        cyCurrentNetwork.elements().remove();
        cyCurrentNetwork.add(elements);
        cyCurrentNetwork.layout(cyConfig.layout).run();
    }
    
    if (cyOptimizedNetwork) {
        const elements = prepareCytoscapeElements('optimized');
        cyOptimizedNetwork.elements().remove();
        cyOptimizedNetwork.add(elements);
        cyOptimizedNetwork.layout(cyConfig.layout).run();
    }
    
    // Mettre à jour les statistiques
    displayNetworkStats();
}

/**
 * Affiche les statistiques de profondeur
 */
function displayDepthStats() {
    if (!depthData) return;
    
    const statsContainer = document.getElementById('depth-stats');
    if (!statsContainer) return;
    
    const stats = depthData.stats;
    const maxDepth = depthData.max_depth;
    
    let statsHtml = `
        <div class="card mt-3">
            <div class="card-header">
                <h5 class="mb-0">Statistiques par niveau de profondeur</h5>
            </div>
            <div class="card-body">
                <p><strong>Profondeur maximale:</strong> ${maxDepth}</p>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Niveau</th>
                                <th>Pages</th>
                                <th>Types de pages</th>
                            </tr>
                        </thead>
                        <tbody>
    `;
    
    // Générer les lignes pour chaque niveau de profondeur
    for (let depth = 0; depth <= maxDepth; depth++) {
        const depthStat = stats[depth] || { count: 0, types: {} };
        
        // Formater les types de pages
        let typesHtml = '';
        for (const [type, count] of Object.entries(depthStat.types)) {
            typesHtml += `<span class="badge bg-secondary me-1">${type}: ${count}</span>`;
        }
        
        statsHtml += `
            <tr>
                <td>${depth === 0 ? 'Racine' : depth}</td>
                <td>${depthStat.count}</td>
                <td>${typesHtml}</td>
            </tr>
        `;
    }
    
    statsHtml += `
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    statsContainer.innerHTML = statsHtml;
}

/**
 * Met à jour les contrôles de l'interface avec les valeurs de la configuration actuelle
 * mais préserve les options qui doivent être activées par défaut
 */
function updateControlsFromConfig() {
    // Mettre à jour le nombre maximum de nœuds
    const maxNodesSlider = document.getElementById('maxNodes');
    const maxNodesValue = document.getElementById('maxNodesValue');
    if (maxNodesSlider && maxNodesValue) {
        maxNodesSlider.value = nodeDisplayConfig.maxNodes;
        maxNodesValue.textContent = nodeDisplayConfig.maxNodes;
    }
    
    // Mettre à jour le select de layout
    const layoutSelect = document.getElementById('layoutSelect');
    if (layoutSelect) {
        // Utiliser le layout par défaut si aucun n'est spécifié
        layoutSelect.value = cyConfig.layout.name || 'cose';
    }
    
    // Mettre à jour l'URL racine
    const rootUrlInput = document.getElementById('rootUrl');
    if (rootUrlInput && visualizationConfig.root_url) {
        rootUrlInput.value = visualizationConfig.root_url;
    }
    
    // Ne pas mettre à jour les options qui doivent être activées par défaut
    // Ces options sont gérées par setupDefaultOptions()
}

/**
 * Configure les événements pour les options avancées
 */
function setupAdvancedOptions() {
    // Configurer l'option de coloration par profondeur
    const colorByDepth = document.getElementById('color-by-depth');
    if (colorByDepth) {
        colorByDepth.addEventListener('change', function() {
            nodeDisplayConfig.colorByDepth = this.checked;
            updateVisualization();
        });
    }
    
    // Configurer l'option de masquage des liens fréquents
    const hideFrequentLinks = document.getElementById('hide-frequent-links');
    if (hideFrequentLinks) {
        hideFrequentLinks.addEventListener('change', function() {
            nodeDisplayConfig.hideFrequentLinks = this.checked;
            updateVisualization();
        });
    }
}

/**
 * Définit l'URL racine par défaut et configure les options initiales
 */
function setupDefaultOptions() {
    // S'assurer que visualizationConfig existe
    if (!visualizationConfig) {
        console.warn('visualizationConfig n\'est pas encore initialisé');
        return;
    }
    
    // Définir l'URL racine par défaut à la page d'accueil
    const rootUrlInput = document.getElementById('rootUrl');
    if (rootUrlInput) {
        try {
            const currentUrl = window.location.href;
            const urlObj = new URL(currentUrl);
            const rootUrl = `${urlObj.protocol}//${urlObj.hostname}`;
            rootUrlInput.value = rootUrl;
        } catch (e) {
            console.error('Erreur lors de la définition de l\'URL racine par défaut:', e);
        }
    }
    
    // Configurer les liens avec l'ID du job
    const backToResultsLink = document.getElementById('back-to-results');
    if (backToResultsLink) {
        const jobId = window.location.pathname.split('/').pop();
        backToResultsLink.href = `/results/${jobId}`;
    }
    
    // S'assurer que les options sont activées par défaut
    const filterFrequentLinksCheckbox = document.getElementById('filterFrequentLinks');
    if (filterFrequentLinksCheckbox) {
        filterFrequentLinksCheckbox.checked = true;
        visualizationConfig.filter_frequent_links = true;
    }
    
    const colorByDepthCheckbox = document.getElementById('color-by-depth');
    if (colorByDepthCheckbox) {
        colorByDepthCheckbox.checked = true;
        nodeDisplayConfig.colorByDepth = true;
    }
    
    const hideFrequentLinksCheckbox = document.getElementById('hide-frequent-links');
    if (hideFrequentLinksCheckbox) {
        hideFrequentLinksCheckbox.checked = true;
        nodeDisplayConfig.hideFrequentLinks = true;
    }
}

// Exporter les fonctions
window.initCytoscapeVisualization = initCytoscapeVisualization;
window.updateCyMaxNodes = updateCyMaxNodes;
window.changeLayout = changeLayout;
window.updateVisualization = updateVisualization;

// Pas d'initialisation automatique au chargement de la page
