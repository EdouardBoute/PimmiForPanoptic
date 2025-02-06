from pydantic import BaseModel
from panoptic.core.plugin.plugin import APlugin
from panoptic.models import Instance, ActionContext
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

from panoptic.core.interfaces import PluginProjectInterface
from panoptic.core.action import ActionContext, ActionResult
import pimmi
import numpy as np

class PimmiParams(BaseModel):
    threshold: float = 0.5
  
class PimmiClusterPlugin(APlugin):  
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):  
        super().__init__(name=PimmiCluster,project=project, plugin_path=plugin_path)  
        self.params = PimmiParams()

        self.add_action_easy(self.cluster_images, ["group"])
  
    async def cluster_images(self, context: ActionContext, threshold: float = 0.5):
        """
        Fonction de clustering basée sur PIMMI.
        :param context: Contexte d'exécution de Panoptic.
        :param threshold: Seuil de similarité pour regrouper les images.
        :return: ActionResult avec les groupes d'images.
        """
        try:
            instance_ids = context.instance_ids  # Liste des ID des images sélectionnées
            images = self.project.get_instances(ids=instance_ids)

            if not images:
                return ActionResult(errors=[{"name": "No images", "message": "Aucune image sélectionnée"}])

            # Extraire les embeddings avec PIMMI
            embeddings = [pimmi.compute_embedding(img.file_path) for img in images]
            clusters = pimmi.cluster_embeddings(np.array(embeddings), threshold=threshold)

            # Créer les groupes pour Panoptic
            grouped_instances = {}
            for img, cluster_id in zip(images, clusters):
                grouped_instances.setdefault(cluster_id, []).append(img.id)

            result_groups = [{"ids": ids, "name": f"Cluster {cluster}"} for cluster, ids in grouped_instances.items()]
            
            return ActionResult(groups=result_groups)

        except Exception as e:
            return ActionResult(errors=[{"name": "Processing Error", "message": str(e)}])